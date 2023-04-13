import Cocoa
import MetalPerformanceShadersGraph

let device = MTLCreateSystemDefaultDevice()!
let commandQueue = device.makeCommandQueue()!

 let expectedWeights: [Float] = [-0.46]
let trainingSamples: [[Float]] = [
    [1.1],
    [0.9],
    [-3.23],
    [4.67],
    [1.44],
    [-2.99],
    [0.82],
    [-2.01],
    [-5.59],
    [4.37]
]
let batchSize: Int = 2
let inputChannels = expectedWeights.count
let outputChannels = 1
let batchIterations = trainingSamples.count / batchSize
var trainingResults: [Float] =  trainingSamples.map {
    $0.enumerated().reduce(0) { $0 + $1.element * expectedWeights[$1.offset] }
}

let graph = MPSGraph()
let batchSizeTensor = graph.constant(Double(batchSize), dataType: .float32)

let inputPlaceholderTensor = graph.placeholder(shape: [batchSize as NSNumber, inputChannels as NSNumber], dataType: .float32, name: "input")
let labelsPlaceholderTensor = graph.placeholder(shape: [batchSize as NSNumber, outputChannels as NSNumber], dataType: .float32, name: "labels")

let weightsValues = (1...inputChannels).map { _ in Float.random(in: -1.0...1.0) }
print("Initial weights: \(weightsValues)")
let weightsData = Data(bytes: weightsValues,
                       count: inputChannels * outputChannels * MemoryLayout<Float>.size)
let weights = graph.variable(with: weightsData,
                             shape: [inputChannels as NSNumber, outputChannels as NSNumber],
                             dataType: .float32,
                             name: "weights")

let outputTensor = graph.matrixMultiplication(primary: inputPlaceholderTensor, secondary: weights, name: "weightedNode")

let subtractionTensor = graph.subtraction(labelsPlaceholderTensor, outputTensor, name: "subtraction")
let squareTensor = graph.square(with: subtractionTensor, name: "square")
let meanSquareLossTensor = graph.mean(of: squareTensor, axes: [0 as NSNumber], name: "lossFunction")
//let meanSquareLossTensor = graph.division(squareTensor, batchSizeTensor, name: "lossFunction")
print(meanSquareLossTensor.debugDescription)

let mpsDevice = MPSGraphDevice(mtlDevice: device)
let sampleCount = trainingSamples.count * inputChannels

let gradientTensors = graph.gradients(of: meanSquareLossTensor, with: [weights], name: "gradientTensor")
let lambdaTensor = graph.constant(0.1, dataType: .float32)
var updateOperations: [MPSGraphOperation] = []
for (key, value) in gradientTensors {
    let updateTensor = graph.stochasticGradientDescent(learningRate: lambdaTensor,
                                                       values: key,
                                                       gradient: value,
                                                       name: nil)
    let assign = graph.assign(key, tensor: updateTensor, name: nil)
    updateOperations += [assign]
}
let executionDescriptor = MPSGraphExecutionDescriptor()
//executionDescriptor.completionHandler = { _, _ in
//}

for _ in 0..<10 {
    for iteration in 0 ..< batchIterations {
        let batchRange = iteration * batchSize ..< (iteration + 1) * batchSize
        let inputDescriptor = MPSNDArrayDescriptor(dataType: .float32,
                                                   shape: [batchSize as NSNumber, inputChannels as NSNumber])
        let inputArray = MPSNDArray(device: device, descriptor: inputDescriptor)
        var batchTrainingSamples = Array(trainingSamples[batchRange].joined())
        inputArray.writeBytes(&batchTrainingSamples, strideBytes: nil)
        let inputTensorData = MPSGraphTensorData(inputArray)
        
        let labelsDescriptor = MPSNDArrayDescriptor(dataType: .float32,
                                                    shape: [batchSize as NSNumber, outputChannels as NSNumber])
        let labelsArray = MPSNDArray(device: device, descriptor: labelsDescriptor)
        var batchLabels = Array(trainingResults[batchRange])
        labelsArray.writeBytes(&batchLabels, strideBytes: nil)
        let labelsTensorData = MPSGraphTensorData(labelsArray)
        
        let commandBuffer = commandQueue.makeCommandBuffer()!
        let fetch = graph.encode(to: MPSCommandBuffer(commandBuffer: commandBuffer),
                                 feeds: [
                                    inputPlaceholderTensor : inputTensorData,
                                    labelsPlaceholderTensor : labelsTensorData
                                 ],
                                 targetTensors: [outputTensor, meanSquareLossTensor, weights],
                                 targetOperations: updateOperations,
                                 executionDescriptor: executionDescriptor)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        let outputTensorData = fetch[outputTensor]!
        let outputNdArray = outputTensorData.mpsndarray()
        var outputs = [Float](repeating: 0, count: batchSize)
        outputNdArray.readBytes(&outputs, strideBytes: nil)
        print("outputs: \(outputs)")
        
        let lossTensorData = fetch[meanSquareLossTensor]!
        let lossNdArray = lossTensorData.mpsndarray()
        var losses = [Float](repeating: 0, count: 1)
        lossNdArray.readBytes(&losses, strideBytes: nil)
        print("losses: \(losses)")
        
        let updatedWeightsTensorData = fetch[weights]!
        var updatedWeights = [Float](repeating: 0, count: inputChannels)
        updatedWeightsTensorData.mpsndarray().readBytes(&updatedWeights, strideBytes: nil)
        print("weights: \(updatedWeights)")
    }
}
