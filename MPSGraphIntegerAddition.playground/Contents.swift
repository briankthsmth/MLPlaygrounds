import Cocoa
import MetalPerformanceShadersGraph

let device = MTLCreateSystemDefaultDevice()!
let commandQueue = device.makeCommandQueue()!

let graph = MPSGraph()
let input1Tensor = graph.placeholder(shape: [1,1], dataType: .int8, name: "input1")
let input2Tensor = graph.placeholder(shape: [1,1], dataType: .int8, name: "input2")
let outputTensor = graph.addition(input1Tensor, input2Tensor, name: "addLayer")

let input1: UInt8 = 2
let input2: UInt8 = 5
let data1 = Data(repeating: input1, count: 1)
let data2 = Data(repeating: input2, count: 1)
let input1TensorData = MPSGraphTensorData(device: MPSGraphDevice(mtlDevice: device),
                                          data: data1,
                                          shape: [1,1],
                                          dataType: .int8)
let input2TensorData = MPSGraphTensorData(device: MPSGraphDevice(mtlDevice: device),
                                          data: data2,
                                          shape: [1,1],
                                          dataType: .int8)
let results = graph.run(with: commandQueue,
                        feeds: [
                            input1Tensor : input1TensorData,
                            input2Tensor : input2TensorData
                        ],
                        targetTensors: [outputTensor],
                        targetOperations: nil)

let outputTensorData = results[outputTensor]!
var result: Int8 = 0
outputTensorData.mpsndarray().readBytes(&result, strideBytes: nil)
print(result)
