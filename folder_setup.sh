git clone https://github.com/PaddlePaddle/DeepSpeech/
cd DeepSpeech/
mkdir checkpoints
cd checkpoints/
mkdir baidu
cd baidu/
mkdir step_final/
cd step_final/
cd .. 
cd .. 
cd ..
cd models/
cd baidu_en8k/
sh download_model.sh
cd ..
cd ..
cp -p models/baidu_en8k/params.pdparams checkpoints/baidu/step_final/params.pdparams
cp -p ../data_utils/audio.py ../DeepSpeech/data_utils/audio.py
cp -p ../data_utils/data.py ../DeepSpeech/data_utils/data.py
cp -p ../data_utils/utility.py ../DeepSpeech/data_utils/utility.py
cp -p ../model_utils/model.py ../DeepSpeech/model_utils/model.py
cp -p ../utils/utility.py ../DeepSpeech/utils/utility.py
cp -p ../utils/error_rate.py ../DeepSpeech/utils/error_rate.py
cp -p ../decoders/swig_wrapper.py ../DeepSpeech/decoders/swig_wrapper.py
cp -p ../swig_decoders.py ../DeepSpeech/swig_decoders.py
cp -p ../_swig_decoders.py ../DeepSpeech/_swig_decoders.py
cp -p ../_swig_decoders.cpython-35m-x86_64-linux-gnu.so ../DeepSpeech/_swig_decoders.cpython-35m-x86_64-linux-gnu.so
cp -p ../create_manifest.py ../DeepSpeech/create_manifest.py
cp -p ../infer_deepspeech2_baidu.py ../DeepSpeech/infer_deepspeech2_baidu.py
cp -p ../models/lm/download_lm_en.sh ../DeepSpeech/models/lm/download_lm_en.sh
cp -p ../deploy/demo_server.py ../DeepSpeech/deploy/demo_server.py
cp -p ../deploy/demo_client.py ../DeepSpeech/deploy/demo_client.py
mkdir test_dataset
cp -r ../test_samples/ ../DeepSpeech/test_dataset/

cd data
mkdir baidu_en8k/
cd ..
cd models/lm
sh download_lm_en.sh
cd ../..
