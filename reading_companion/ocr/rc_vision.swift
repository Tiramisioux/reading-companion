// rc-vision: Apple Vision text recognition for reading-companion.
//
//   rc-vision [--languages en-US,sv-SE] FILE...
//
// Prints one JSON line per input file, in input order:
//   {"file": "<path>", "width": W, "height": H,
//    "lines": [{"text": "...", "conf": 0.98, "x": 0.1, "y": 0.2, "w": 0.3, "h": 0.05,
//               "words": [{"text": "...", "x": 0.1, "y": 0.2, "w": 0.05, "h": 0.05}]}]}
// "words" are the line's whitespace-separated tokens with Vision's own box for each (a word is
// left out when Vision gives no box for it). Coordinates are normalised 0..1 to the image with the
// origin at the TOP-LEFT (Vision's own origin is bottom-left; this tool converts). A file that
// cannot be read gives {"file", "error"}.
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

func box(_ b: CGRect) -> [String: Any] {
    return ["x": Double(b.origin.x), "y": Double(1.0 - b.origin.y - b.height), "w": Double(b.width), "h": Double(b.height)]
}

func wordBoxes(_ c: VNRecognizedText) -> [[String: Any]] {
    let s = c.string
    var out: [[String: Any]] = []
    var i = s.startIndex
    while i < s.endIndex {
        while i < s.endIndex && s[i].isWhitespace { i = s.index(after: i) }
        if i >= s.endIndex { break }
        var j = i
        while j < s.endIndex && !s[j].isWhitespace { j = s.index(after: j) }
        if let r = try? c.boundingBox(for: i..<j) {
            var w = box(r.boundingBox)
            w["text"] = String(s[i..<j])
            out.append(w)
        }
        i = j
    }
    return out
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
        var line = box(obs.boundingBox)
        line["text"] = c.string
        line["conf"] = Double(c.confidence)
        line["words"] = wordBoxes(c)
        lines.append(line)
    }
    emit(["file": path, "width": cg.width, "height": cg.height, "lines": lines])
}
