import Cocoa
import MLCompute

let inputTensor = MLCTensor(shape: [2,4], dataType: .float32)
let weightTensor = MLCTensor(shape: [1,8], fillWithData: 1.0, dataType: .float32)

let graph = MLCGraph()

let fullyConnectedDescriptor = MLCConvolutionDescriptor(kernelSizes: (4, 2), inputFeatureChannelCount: 4, outputFeatureChannelCount: 2)
let fullyConnectedLayer = MLCFullyConnectedLayer(weights: weightTensor, biases: nil, descriptor: fullyConnectedDescriptor)

let outputTensor = graph.node(with: fullyConnectedLayer!, source: inputTensor)

let inferenceGraph = MLCInferenceGraph(graphObjects: [graph])
inferenceGraph.addInputs(["inTensor" : inputTensor])
inferenceGraph.compile(device: MLCDevice.cpu())


var inputs: [Float] = [0,1,0,1,1, 1, 1, 1]
let inputCount = inputs.count
let inputTensorData = inputs.withUnsafeMutableBufferPointer { pointer in
    MLCTensorData(bytesNoCopy: pointer.baseAddress!, length: inputCount * MemoryLayout<Float>.size)
}
inferenceGraph.execute(inputsData: ["inTensor" : inputTensorData], batchSize: 2, options: [.synchronous])

var outputs = [Float](repeating: 0, count: 4)
outputTensor?.copyDataFromDeviceMemory(toBytes: &outputs, length: outputs.count * MemoryLayout<Float>.size, synchronizeWithDevice: false)
outputs.debugDescription
