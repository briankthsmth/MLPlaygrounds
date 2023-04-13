import Cocoa
import MLCompute

let inputArray = [[[[Float]]]]([[
    [[0.1, 0.1, 0.1], [0.2, 0.2, 0.2], [0.3, 0.3, 0.3]],
    [[0.4, 0.4, 0.4], [0.5, 0.5, 0.5], [0.6, 0.6, 0.6]]
]])
let inputVector = Array(inputArray.joined().joined().joined())
inputVector.debugDescription
let numberOfBytes = inputVector.count * MemoryLayout<Float>.size
let inputTensorData = inputVector.withUnsafeBytes { pointer in
    MLCTensorData(immutableBytesNoCopy: pointer.baseAddress!, length: numberOfBytes)
}

let graph = MLCGraph()
let inputTensor = MLCTensor(shape: [1, 2, 3, 3], dataType: .float32)

let transposeLayer = MLCTransposeLayer(dimensions: [0, 3, 1, 2])!
let output = graph.node(with: transposeLayer, source: inputTensor)!

let inferenceGraph = MLCInferenceGraph(graphObjects: [graph])
inferenceGraph.addInputs(["inputTensor" : inputTensor])

inferenceGraph
    .compile(device: .cpu())

inferenceGraph
    .execute(inputsData: ["inputTensor" : inputTensorData], batchSize: 1, options: [.synchronous])

var outputVector = [Float](repeating: 0, count: inputVector.count)
output.copyDataFromDeviceMemory(toBytes: &outputVector, length: numberOfBytes, synchronizeWithDevice: false)
outputVector.debugDescription

// restore

let reverseInputData = outputVector.withUnsafeBytes { pointer in
    MLCTensorData(immutableBytesNoCopy: pointer.baseAddress!, length: numberOfBytes)
}
let reverseGraph = MLCGraph()
let reverseInputTensor =  MLCTensor(shape: [1, 3, 2, 3], dataType: .float32)
let reverseTransposeLayer = MLCTransposeLayer(dimensions: [0, 2, 3, 1])!
let reverseOutput = reverseGraph.node(with: reverseTransposeLayer, source: reverseInputTensor)!

let reverseInferenceGraph = MLCInferenceGraph(graphObjects: [reverseGraph])
reverseInferenceGraph.addInputs(["inputTensor" : reverseInputTensor])

reverseInferenceGraph
    .compile(device: .cpu())
reverseInferenceGraph
    .execute(inputsData: ["inputTensor" : reverseInputData], batchSize: 1, options: [.synchronous])

var reverseOutputVector = [Float](repeating: 0, count: inputVector.count)
reverseOutput.copyDataFromDeviceMemory(toBytes: &reverseOutputVector, length: numberOfBytes, synchronizeWithDevice: false)
reverseOutputVector.debugDescription
