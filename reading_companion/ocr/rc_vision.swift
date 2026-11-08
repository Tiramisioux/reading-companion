// rc-vision: Apple Vision text recognition for reading-companion.
//
//   rc-vision [--languages en-US,sv-SE] FILE...
//
// Prints one JSON line per input file, in input order:
//   {"file": "<path>", "width": W, "height": H,
//    "lines": [{"text": "...", "conf": 0.98, "x": 0.1, "y": 0.2, "w": 0.3, "h": 0.05}]}
// Coordinates are normalised 0..1 to the image with the origin at the TOP-LEFT (Vision's own
// origin is bottom-left; this tool converts). A file that cannot be read gives {"file", "error"}.
import AppKit
import Foundation
import Vision

var languages = ["en-US", "sv-SE"]
var files: [String] = []
var args = Array(CommandLine.arguments.dropFirst())
while !args.isEmpty {
    let a = args.removeFirst()
    if a == "--languages", !args.isEmpty {
        languages = args.removeFirst().split(separator: ",").map { String($0) }
    } else {
        files.append(a)
    }
}

func emit(_ obj: [String: Any]) {
    if let d = try? JSONSerialization.data(withJSONObject: obj), let s = String(data: d, encoding: .utf8) {
        print(s)
    }
}

for path in files {
    guard let img = NSImage(contentsOfFile: path),
          let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
        emit(["file": path, "error": "cannot read image"])
        continue
    }
    let req = VNRecognizeTextRequest()
    req.recognitionLevel = .accurate
    req.usesLanguageCorrection = false
    req.recognitionLanguages = languages
    let handler = VNImageRequestHandler(cgImage: cg, options: [:])
    do {
        try handler.perform([req])
    } catch {
        emit(["file": path, "error": "\(error)"])
        continue
    }
    var lines: [[String: Any]] = []
    for obs in req.results ?? [] {
        guard let c = obs.topCandidates(1).first else { continue }
        let b = obs.boundingBox
        lines.append(["text": c.string, "conf": Double(c.confidence),
                      "x": Double(b.origin.x), "y": Double(1.0 - b.origin.y - b.height),
                      "w": Double(b.width), "h": Double(b.height)])
    }
    emit(["file": path, "width": cg.width, "height": cg.height, "lines": lines])
}
