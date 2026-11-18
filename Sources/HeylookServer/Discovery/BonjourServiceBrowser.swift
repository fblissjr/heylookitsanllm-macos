import Foundation
import Network

/// Represents a `heylook` server discovered via Bonjour (`_heylook._tcp`) or configured over LAN/Tailscale.
public struct DiscoveredHeylookServer: Identifiable, Sendable, Equatable, Codable {
    public let id: String
    public let name: String
    public let domain: String
    public let baseURLString: String
    public let isBonjour: Bool

    public init(
        id: String,
        name: String,
        domain: String = "local.",
        baseURLString: String,
        isBonjour: Bool = true
    ) {
        self.id = id
        self.name = name
        self.domain = domain
        self.baseURLString = baseURLString
        self.isBonjour = isBonjour
    }

    /// Derives a connectable HTTP base URL from a Bonjour service name or hostname and port.
    public static func fromBonjourService(
        name: String,
        type: String = "_heylook._tcp",
        domain: String = "local.",
        defaultPort: Int = 8000
    ) -> DiscoveredHeylookServer {
        let sanitizedHost = name
            .replacingOccurrences(of: " ", with: "-")
            .replacingOccurrences(of: "'", with: "")
            .replacingOccurrences(of: "(", with: "")
            .replacingOccurrences(of: ")", with: "")
        let hostWithDomain = domain == "local." || domain == "local"
            ? "\(sanitizedHost).local"
            : "\(sanitizedHost).\(domain.trimmingCharacters(in: CharacterSet(charactersIn: ".")))"
        let url = "http://\(hostWithDomain):\(defaultPort)"
        return DiscoveredHeylookServer(
            id: "\(name).\(type).\(domain)",
            name: name,
            domain: domain,
            baseURLString: url,
            isBonjour: true
        )
    }
}

/// Discovers embedded `heylook` servers advertising `_heylook._tcp` via `Network.framework` `NWBrowser`.
public final class BonjourServiceBrowser: @unchecked Sendable {
    public static let serviceType = "_heylook._tcp"

    private let queue = DispatchQueue(label: "com.heylook.bonjour.browser")
    private var browser: NWBrowser?
    private let defaultPort: Int

    public init(defaultPort: Int = 8000) {
        self.defaultPort = defaultPort
    }

    public func start(onUpdate: @escaping @Sendable ([DiscoveredHeylookServer]) -> Void) {
        stop()
        let descriptor = NWBrowser.Descriptor.bonjour(type: Self.serviceType, domain: nil)
        let params = NWParameters.tcp
        params.includePeerToPeer = true
        let nwBrowser = NWBrowser(for: descriptor, using: params)
        let port = self.defaultPort
        nwBrowser.browseResultsChangedHandler = { results, _ in
            var servers: [DiscoveredHeylookServer] = []
            for result in results {
                if case let .service(name, type, domain, _) = result.endpoint {
                    servers.append(
                        DiscoveredHeylookServer.fromBonjourService(
                            name: name,
                            type: type,
                            domain: domain,
                            defaultPort: port
                        )
                    )
                }
            }
            servers.sort { $0.name.localizedStandardCompare($1.name) == .orderedAscending }
            onUpdate(servers)
        }
        nwBrowser.start(queue: queue)
        self.browser = nwBrowser
    }

    public func stop() {
        browser?.cancel()
        browser = nil
    }
}
