import Cocoa
import MLCompute

//let inputTensor = MLCTensor(shape: [1,3,2,3], dataType: .float32)
let inputDescriptor = MLCTensorDescriptor(width: 3, height: 2, featureChannelCount: 3, batchSize: 1, dataType: .float32)
inputDescriptor.debugDescription
let inputTensor = MLCTensor(descriptor: inputDescriptor!)

var weights: [Float] = [
    /* output channel 0 */
    /* input channel 0 */
    0.5, 0.5,
    0.1, 0.1,
    
    /* input channel 1 */
    1.0, 1.0,
    0.3, 0.3,
    
    /* input channel 2 */
    0.25, 0.25,
    0.5, 0.5,
    
    /* output channel 1 */
    /* input channel 0 */
    0.25, 0.25,
    0.5, 0.5,
    
    /* input channel 1 */
    0.5, 0.5,
    0.6, 0.6,
    
    /* input channel 2 */
    1.0, 1.0,
    0.45, 0.45
]

let weightDescriptor = MLCTensorDescriptor(convolutionWeightsWithWidth: 2,
                                           height: 2,
                                           inputFeatureChannelCount: 3,
                                           outputFeatureChannelCount: 2,
                                           dataType: .float32)!
weightDescriptor.debugDescription

let weightTensor = MLCTensor(descriptor: weightDescriptor,
                             data: MLCTensorData(bytesNoCopy: &weights,
                                                 length: weights.count * MemoryLayout<Float>.size))

let graph = MLCGraph()

let convolutionDescriptor = MLCConvolutionDescriptor(type: .standard , kernelSizes: (2, 2), inputFeatureChannelCount: 3, outputFeatureChannelCount: 2)
let layer = MLCConvolutionLayer(weights: weightTensor, biases: nil, descriptor: convolutionDescriptor)

let outputTensor = graph.node(with: layer!, source: inputTensor)
outputTensor?.descriptor.debugDescription
let outputSize = outputTensor!.descriptor.tensorAllocationSizeInBytes

let inferenceGraph = MLCInferenceGraph(graphObjects: [graph])
inferenceGraph.addInputs(["inTensor" : inputTensor])
inferenceGraph.compile(device: MLCDevice.gpu()!)


var inputs: [Float] = [
    /* feature channel 0 */
    0.1, 0.1, 0.1,
    0.2, 0.2, 0.2,
    /* feature channel 1 */
    0.3, 0.3, 0.3,
    0.4, 0.4, 0.4,
    /* feature channel 2 */
    0.5, 0.5, 0.5,
    0.6, 0.6, 0.6
]
let inputCount = inputs.count
let inputTensorData = inputs.withUnsafeMutableBufferPointer { pointer in
    MLCTensorData(bytesNoCopy: pointer.baseAddress!, length: inputCount * MemoryLayout<Float>.size)
}
inferenceGraph.execute(inputsData: ["inTensor" : inputTensorData], batchSize: 1, options: [.synchronous])

var outputs = [Float](repeating: 0, count: outputSize / MemoryLayout<Float>.size)
outputTensor?.copyDataFromDeviceMemory(toBytes: &outputs, length: outputSize, synchronizeWithDevice: false)
outputs.debugDescription
