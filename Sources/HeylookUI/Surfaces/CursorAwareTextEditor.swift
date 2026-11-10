import Foundation
import SwiftUI

#if os(macOS)
import AppKit

/// Multiplatform cursor- and selection-aware plain-text editor (`NSTextView` on macOS, `UITextView` on iOS)
/// enabling true inline mid-buffer Notebook continuation, selection extraction, Undo/Redo, and
/// read-only locking while tokens stream at the insertion point.
public struct CursorAwareTextEditor: NSViewRepresentable {
    @Binding public var text: String
    @Binding public var selectedRange: NSRange
    public var isEditable: Bool
    public var onUserEdit: (() -> Void)?

    public init(
        text: Binding<String>,
        selectedRange: Binding<NSRange>,
        isEditable: Bool = true,
        onUserEdit: (() -> Void)? = nil
    ) {
        self._text = text
        self._selectedRange = selectedRange
        self.isEditable = isEditable
        self.onUserEdit = onUserEdit
    }

    public func makeCoordinator() -> Coordinator {
        Coordinator(parent: self)
    }

    public func makeNSView(context: Context) -> NSScrollView {
        let scrollView = NSTextView.scrollableTextView()
        scrollView.drawsBackground = false
        scrollView.hasVerticalScroller = true
        scrollView.hasHorizontalScroller = false
        scrollView.autohidesScrollers = true
        scrollView.borderType = .noBorder

        if let textView = scrollView.documentView as? NSTextView {
            textView.delegate = context.coordinator
            textView.isRichText = false
            textView.importsGraphics = false
            textView.allowsUndo = true
            textView.usesFindBar = true
            textView.isIncrementalSearchingEnabled = true
            textView.isAutomaticQuoteSubstitutionEnabled = false
            textView.isAutomaticDashSubstitutionEnabled = false
            textView.isAutomaticTextReplacementEnabled = false
            textView.drawsBackground = false
            textView.font = NSFont.monospacedSystemFont(ofSize: 13.5, weight: .regular)
            textView.textColor = .labelColor
            textView.insertionPointColor = .controlAccentColor
            textView.textContainerInset = NSSize(width: 24, height: 18)
            textView.isEditable = isEditable
            textView.isSelectable = true

            context.coordinator.isUpdatingProgrammatically = true
            textView.string = text
            let clamped = Self.clampRange(selectedRange, utf16Length: (text as NSString).length)
            textView.setSelectedRange(clamped)
            context.coordinator.isUpdatingProgrammatically = false
        }

        return scrollView
    }

    public func updateNSView(_ scrollView: NSScrollView, context: Context) {
        context.coordinator.parent = self
        guard let textView = scrollView.documentView as? NSTextView else { return }

        context.coordinator.isUpdatingProgrammatically = true
        defer { context.coordinator.isUpdatingProgrammatically = false }

        if textView.isEditable != isEditable {
            textView.isEditable = isEditable
        }

        let targetLength = (text as NSString).length
        let clamped = Self.clampRange(selectedRange, utf16Length: targetLength)

        if textView.string != text {
            textView.string = text
            textView.setSelectedRange(clamped)
            if !isEditable {
                textView.scrollRangeToVisible(clamped)
            }
        } else if textView.selectedRange() != clamped {
            textView.setSelectedRange(clamped)
        }
    }

    static func clampRange(_ range: NSRange, utf16Length: Int) -> NSRange {
        let safeLoc = min(max(0, range.location), utf16Length)
        let maxLen = max(0, utf16Length - safeLoc)
        let safeLen = min(max(0, range.length), maxLen)
        return NSRange(location: safeLoc, length: safeLen)
    }

    @MainActor
    public final class Coordinator: NSObject, NSTextViewDelegate {
        var parent: CursorAwareTextEditor
        var isUpdatingProgrammatically: Bool = false

        init(parent: CursorAwareTextEditor) {
            self.parent = parent
        }

        public func textDidChange(_ notification: Notification) {
            guard !isUpdatingProgrammatically,
                  let textView = notification.object as? NSTextView else { return }
            let updatedText = textView.string
            let updatedRange = textView.selectedRange()
            if parent.text != updatedText {
                parent.text = updatedText
            }
            if parent.selectedRange != updatedRange {
                parent.selectedRange = updatedRange
            }
            parent.onUserEdit?()
        }

        public func textViewDidChangeSelection(_ notification: Notification) {
            guard !isUpdatingProgrammatically,
                  let textView = notification.object as? NSTextView else { return }
            let newRange = textView.selectedRange()
            guard parent.selectedRange != newRange else { return }
            Task { @MainActor [weak self] in
                guard let self, !self.isUpdatingProgrammatically else { return }
                if self.parent.selectedRange != newRange {
                    self.parent.selectedRange = newRange
                }
            }
        }
    }
}

#elseif canImport(UIKit)
import UIKit

public struct CursorAwareTextEditor: UIViewRepresentable {
    @Binding public var text: String
    @Binding public var selectedRange: NSRange
    public var isEditable: Bool
    public var onUserEdit: (() -> Void)?

    public init(
        text: Binding<String>,
        selectedRange: Binding<NSRange>,
        isEditable: Bool = true,
        onUserEdit: (() -> Void)? = nil
    ) {
        self._text = text
        self._selectedRange = selectedRange
        self.isEditable = isEditable
        self.onUserEdit = onUserEdit
    }

    public func makeCoordinator() -> Coordinator {
        Coordinator(parent: self)
    }

    public func makeUIView(context: Context) -> UITextView {
        let textView = UITextView()
        textView.delegate = context.coordinator
        textView.backgroundColor = .clear
        textView.font = UIFont.monospacedSystemFont(ofSize: 14, weight: .regular)
        textView.textColor = .label
        textView.autocorrectionType = .no
        textView.smartQuotesType = .no
        textView.smartDashesType = .no
        textView.textContainerInset = UIEdgeInsets(top: 16, left: 16, bottom: 16, right: 16)
        textView.isEditable = isEditable
        textView.isSelectable = true

        context.coordinator.isUpdatingProgrammatically = true
        textView.text = text
        let clamped = Self.clampRange(selectedRange, utf16Length: (text as NSString).length)
        textView.selectedRange = clamped
        context.coordinator.isUpdatingProgrammatically = false
        return textView
    }

    public func updateUIView(_ textView: UITextView, context: Context) {
        context.coordinator.parent = self
        context.coordinator.isUpdatingProgrammatically = true
        defer { context.coordinator.isUpdatingProgrammatically = false }

        if textView.isEditable != isEditable {
            textView.isEditable = isEditable
        }

        let targetLength = (text as NSString).length
        let clamped = Self.clampRange(selectedRange, utf16Length: targetLength)

        if textView.text != text {
            textView.text = text
            textView.selectedRange = clamped
            if !isEditable {
                textView.scrollRangeToVisible(clamped)
            }
        } else if textView.selectedRange != clamped {
            textView.selectedRange = clamped
        }
    }

    static func clampRange(_ range: NSRange, utf16Length: Int) -> NSRange {
        let safeLoc = min(max(0, range.location), utf16Length)
        let maxLen = max(0, utf16Length - safeLoc)
        let safeLen = min(max(0, range.length), maxLen)
        return NSRange(location: safeLoc, length: safeLen)
    }

    @MainActor
    public final class Coordinator: NSObject, UITextViewDelegate {
        var parent: CursorAwareTextEditor
        var isUpdatingProgrammatically: Bool = false

        init(parent: CursorAwareTextEditor) {
            self.parent = parent
        }

        public func textViewDidChange(_ textView: UITextView) {
            guard !isUpdatingProgrammatically else { return }
            let updatedText = textView.text ?? ""
            let updatedRange = textView.selectedRange
            if parent.text != updatedText {
                parent.text = updatedText
            }
            if parent.selectedRange != updatedRange {
                parent.selectedRange = updatedRange
            }
            parent.onUserEdit?()
        }

        public func textViewDidChangeSelection(_ textView: UITextView) {
            guard !isUpdatingProgrammatically else { return }
            let newRange = textView.selectedRange
            guard parent.selectedRange != newRange else { return }
            Task { @MainActor [weak self] in
                guard let self, !self.isUpdatingProgrammatically else { return }
                if self.parent.selectedRange != newRange {
                    self.parent.selectedRange = newRange
                }
            }
        }
    }
}
#endif
