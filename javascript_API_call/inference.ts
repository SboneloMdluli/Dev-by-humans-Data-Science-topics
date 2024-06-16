import * as onnxruntime from 'onnxruntime-node';
import {data} from './arr'
const mode_path = '../model_storage/boost_clf.onnx';


async function predict(path: string, array: Float32Array, dims: number[]): Promise<[any, number]> {
  try {
    const onnxSession = await onnxruntime.InferenceSession.create(path);
    const inputTensor = new onnxruntime.Tensor('float32', array, dims);

    var [results, inferenceDuration] = await runInference(onnxSession, inputTensor);
    return results;
  } catch (error) {
    console.log(error);
    return [[], 0];
  }
}


async function runInference(session: any, data: any): Promise<[any, number]> {
  const start = new Date();
  const feeds: Record<string, any> = {};
  feeds[session.inputNames[0]] = data;
  // Run the session inference.
  const outputData = await session.run(feeds);
  const end = new Date();
  const inferenceDuration = (end.getTime() - start.getTime()) / 1000;
  return [outputData, inferenceDuration];
}


data().forEach(customer => {
  const inputTempArray = Float32Array.from(customer);
  const dims = [1,inputTempArray.length]
  predict(mode_path,inputTempArray,dims).then(function(result) {
    console.log(result['label']);
  });
});
