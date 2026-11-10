import Darwin
import Foundation
import HeylookCore

/// Re-export `HeylookCore.GGUFHeaderReader` for callers importing `HeylookEngines`.
public typealias GGUFHeaderReader = HeylookCore.GGUFHeaderReader

/// Dynamic conflict-free loopback port pool (`18080+`) for co-resident `llama-server` (`GGUF`)
/// subprocesses so multiple `.gguf` models up to `maxLoadedGPUModels` never collide on a single port
/// or collide with external OS listeners already bound to `127.0.0.1`.
public final class GGUFPortPool: @unchecked Sendable {
    public static let shared = GGUFPortPool(basePort: 18080)

    public let basePort: Int
    private let lock = NSLock()
    private var assignedPorts: [String: Int] = [:]

    public init(basePort: Int = 18080) {
        self.basePort = basePort
    }

    /// Probes whether `host:port` is currently free on the OS TCP stack using `Darwin.socket` + `Darwin.bind`.
    public static func isPortAvailable(_ port: Int, host: String = "127.0.0.1") -> Bool {
        guard port > 0 && port <= 65535 else { return false }
        let fd = Darwin.socket(AF_INET, SOCK_STREAM, 0)
        guard fd >= 0 else { return true }
        defer { Darwin.close(fd) }

        var addr = sockaddr_in()
        addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = in_port_t(UInt16(port).bigEndian)
        let converted = host.withCString { cstr in
            inet_pton(AF_INET, cstr, &addr.sin_addr)
        }
        if converted != 1 {
            addr.sin_addr.s_addr = CFSwapInt32HostToBig(INADDR_LOOPBACK)
        }

        let bindResult = withUnsafePointer(to: &addr) { ptr in
            ptr.withMemoryRebound(to: sockaddr.self, capacity: 1) { sockPtr in
                Darwin.bind(fd, sockPtr, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        return bindResult == 0
    }

    public func acquirePort(for modelID: String) -> Int {
        lock.lock()
        defer { lock.unlock() }
        if let existing = assignedPorts[modelID] {
            return existing
        }
        let used = Set(assignedPorts.values)
        var candidate = basePort
        let maxCandidate = min(65535, basePort + 2048)
        while candidate <= maxCandidate && (used.contains(candidate) || !Self.isPortAvailable(candidate)) {
            candidate += 1
        }
        assignedPorts[modelID] = candidate
        return candidate
    }

    public func assignedPort(for modelID: String) -> Int? {
        lock.lock()
        defer { lock.unlock() }
        return assignedPorts[modelID]
    }

    public func activeAllocations() -> [String: Int] {
        lock.lock()
        defer { lock.unlock() }
        return assignedPorts
    }

    public func releasePort(for modelID: String) {
        lock.lock()
        assignedPorts.removeValue(forKey: modelID)
        lock.unlock()
    }

    public func releaseAll() {
        lock.lock()
        assignedPorts.removeAll()
        lock.unlock()
    }
}

#if os(macOS)
/// Process-wide lifecycle registry for out-of-process child servers (`llama-server`).
/// Ensures that unloading a model, LRU eviction, stopping `HeylookHTTPServer`, or process exit
/// (`atexit`) deterministically terminates (`SIGTERM` -> `SIGKILL`) all spawned child processes
/// so no orphaned `llama-server` processes or leaked unified memory remain.
public final class SubprocessLifecycleRegistry: @unchecked Sendable {
    public static let shared = SubprocessLifecycleRegistry()

    private let lock = NSLock()
    private var processes: [String: Process] = [:]

    private init() {
        atexit {
            SubprocessLifecycleRegistry.shared.terminateAll()
        }
    }

    public func register(_ process: Process, for modelID: String) {
        var oldProcess: Process?
        lock.lock()
        oldProcess = processes[modelID]
        processes[modelID] = process
        lock.unlock()

        if let oldProcess, oldProcess !== process {
            Self.terminateProcess(oldProcess)
        }
    }

    public func terminate(for modelID: String) {
        let proc: Process?
        lock.lock()
        proc = processes.removeValue(forKey: modelID)
        lock.unlock()
        GGUFPortPool.shared.releasePort(for: modelID)

        if let proc {
            Self.terminateProcess(proc)
        }
    }

    public func terminateAll() {
        let allProcs: [Process]
        lock.lock()
        allProcs = Array(processes.values)
        processes.removeAll()
        lock.unlock()
        GGUFPortPool.shared.releaseAll()

        for proc in allProcs {
            Self.terminateProcess(proc)
        }
    }

    public func isRunning(for modelID: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        guard let proc = processes[modelID] else { return false }
        return proc.isRunning
    }

    public func processIdentifier(for modelID: String) -> Int32? {
        lock.lock()
        defer { lock.unlock() }
        guard let proc = processes[modelID], proc.isRunning else { return nil }
        return proc.processIdentifier
    }

    public var activeSubprocessCount: Int {
        lock.lock()
        defer { lock.unlock() }
        return processes.values.filter(\.isRunning).count
    }

    private static func terminateProcess(_ proc: Process) {
        guard proc.isRunning else { return }
        let pid = proc.processIdentifier
        proc.terminate()
        for _ in 0..<10 {
            if !proc.isRunning {
                break
            }
            usleep(20_000)
        }
        if proc.isRunning && pid > 0 {
            kill(pid, SIGKILL)
        }
        proc.waitUntilExit()
    }
}
#endif
