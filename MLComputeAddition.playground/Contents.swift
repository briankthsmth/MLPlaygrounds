import Foundation
import MLCompute

func printData(_ data: Data) {
    data.forEach {
        print($0)
    }
}

let modelGraph = MLCGraph()

let firstInputTensor = MLCTensor(shape: [1, 3], dataType: .float32)
let secondInputTensor = MLCTensor(shape: [1, 3], dataType: .float32)

let firstAddLayer = MLCArithmeticLayer(operation: .add)

let firstOutputTensor = modelGraph.node(with: firstAddLayer,
                                        sources: [firstInputTensor, secondInputTensor])!

let thirdInputTensor = MLCTensor(shape: [1, 3], dataType: .float32)
let secondAddLayer = MLCArithmeticLayer(operation: .add)

let secondOutputTensor = modelGraph.node(with: secondAddLayer,
                                         sources: [firstOutputTensor, thirdInputTensor])!


let inferenceGraph = MLCInferenceGraph(graphObjects: [modelGraph])
inferenceGraph.addInputs([
    "data0" : firstInputTensor,
    "data1" : secondInputTensor,
    "data2" : thirdInputTensor
])

inferenceGraph.compile(device: .gpu()!)

var firstInputVector: [Float] = [1, 1, 1]
var firstInputData = Data(bytes: &firstInputVector, count: 12)
let firstInputTensorData = firstInputData.withUnsafeBytes { pointer in
    let data = MLCTensorData(immutableBytesNoCopy: pointer.baseAddress!,
                  length: 3 * MemoryLayout<Float>.size)
    let tensorData = Data(bytes: data.bytes, count: 12)
    printData(tensorData)
    return data
}


var secondInputVector: [Float] = [2, 2, 2]
var secondInputData = Data(bytes: &secondInputVector, count: 12)
let secondInputTensorData = secondInputData.withUnsafeBytes { pointer in
    MLCTensorData(immutableBytesNoCopy: pointer.baseAddress!,
                  length: 3 * MemoryLayout<Float>.size)
}

var thirdInputVector: [Float] = [3, 3, 3]
var thirdInputData = Data(bytes: &thirdInputVector, count: 12)
let thirdInputTensorData = thirdInputData.withUnsafeBytes { pointer in
    MLCTensorData(immutableBytesNoCopy: pointer.baseAddress!,
                  length: 3 * MemoryLayout<Float>.size)
}

inferenceGraph.execute(inputsData: [
    "data0" : firstInputTensorData,
    "data1" : secondInputTensorData,
    "data2" : thirdInputTensorData
],
                       batchSize: 1)

var outputData = Data(repeating: 0, count: 3 * MemoryLayout<Float>.size)

let _ = outputData.withUnsafeMutableBytes { pointer in
    secondOutputTensor.copyDataFromDeviceMemory(toBytes: pointer.baseAddress!,
                                                length: 3 * MemoryLayout<Float>.size,
                                                synchronizeWithDevice: true)
}


