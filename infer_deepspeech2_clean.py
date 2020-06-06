"""Inferer for DeepSpeech2 model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')

import functools
import paddle.fluid as fluid
from data_utils.data import DataGenerator
from model_utils.model import DeepSpeech2Model
from model_utils.model_check import check_cuda, check_version
from utils.error_rate import wer, cer
from utils.utility import add_arguments, print_arguments
import create_manifest 
import json
import codecs
import soundfile
import time

ds2_model = None
data_generator = None
vocab_list = None

ROOT_PATH = "/home/Nishchith"

num_samples = 20000
beam_size = 500
num_proc_bsearch = 8
num_conv_layers = 2
num_rnn_layers = 3
rnn_layer_size = 1024
alpha = 2.5
beta = 0.3
cutoff_prob = 1.0
cutoff_top_n = 40
use_gru = True
use_gpu = False
share_rnn_weights = False
target_dir = "test_dataset/"
infer_manifest = "data/baidu_en8k/manifest.dev-clean"
mean_std_path = "models/baidu_en8k/mean_std.npz"
vocab_path = "models/baidu_en8k/vocab.txt"
lang_model_path = "models/lm/common_crawl_00.prune01111.trie.klm"
model_path = "checkpoints/baidu/step_final"
error_rate_type = "wer"
specgram_type = "linear"
# audio_path = ""
decoding_method = None # choices = ['ctc_beam_search', 'ctc_greedy']

def prepare_manifest():
    print("Preparing Manifest")
    create_manifest.prepare_dataset(target_dir=target_dir, manifest_path=infer_manifest)


def load_model(_decoding_method="ctc_beam_search"): 
    global decoding_method

    decoding_method = _decoding_method

    print("Decoding Method: ", decoding_method)

    # check if set use_gpu=True in paddlepaddle cpu version
    check_cuda(use_gpu)
    # check if paddlepaddle version is satisfied
    check_version()

    if use_gpu:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()
    
    # Load model
    data_generator = DataGenerator(
        vocab_filepath=vocab_path,
        mean_std_filepath=mean_std_path,
        augmentation_config='{}',
        specgram_type=specgram_type,
        keep_transcription_text=True,
        place = place,
        is_training = False)
    
    ds2_model = DeepSpeech2Model(
        vocab_size=data_generator.vocab_size,
        num_conv_layers=num_conv_layers,
        num_rnn_layers=num_rnn_layers,
        rnn_layer_size=rnn_layer_size,
        use_gru=use_gru,
        share_rnn_weights=share_rnn_weights,
        place=place,
        init_from_pretrained_model=model_path)

    # decoders only accept string encoded in utf-8
    vocab_list = data_generator.vocab_list

    if decoding_method == "ctc_beam_search":
        print("Model loaded partially, waiting for external scorer..")
        ds2_model.init_ext_scorer(alpha, beta, lang_model_path, vocab_list)

    return ds2_model, data_generator, vocab_list
    
    
def infer(ds2_model, data_generator, vocab_list, audio_path):
    """Inference for DeepSpeech2."""
    
    # Prepare manifest
    if audio_path:
        json_lines = []
        audio_data, samplerate = soundfile.read(audio_path)
        duration = float(len(audio_data)) / samplerate
        json_lines.append(
                    json.dumps({
                        'audio_filepath': audio_path,
                        'duration': duration,
                        'text': 'NO TRANSCRIPT'
                    }))
        with codecs.open(infer_manifest, 'w', 'utf-8') as out_file:
            for line in json_lines:
                out_file.write(line + '\n')
    else:
        prepare_manifest()

    # Load audio
    batch_reader = data_generator.batch_reader_creator(
        manifest_path=infer_manifest,
        batch_size=num_samples,
        sortagrad=False,
        shuffle_method=None)
    infer_data = next(batch_reader()) # (padded_audios, texts, audio_lens, masks, audio_file_path)

    ds2_model.logger.info("start inference ...")
    probs_split= ds2_model.infer_batch_probs(
        infer_data=infer_data,
        feeding_dict=data_generator.feeding)

    result_transcripts = ""
    tik = time.time()

    if decoding_method == "ctc_greedy":
        result_transcripts = ds2_model.decode_batch_greedy(
            probs_split=probs_split,
            vocab_list=vocab_list)
    else:
        result_transcripts = ds2_model.decode_batch_beam_search(
            probs_split=probs_split,
            beam_alpha=alpha,
            beam_beta=beta,
            beam_size=beam_size,
            cutoff_prob=cutoff_prob,
            cutoff_top_n=cutoff_top_n,
            vocab_list=vocab_list,
            num_processes=num_proc_bsearch)

    print("Inference Time: ", time.time() - tik)
    ds2_model.logger.info("finish inference")
    print(result_transcripts)

    if len(result_transcripts) == 0:
        return "No transcript available"

    transcript = result_transcripts[0]
    return transcript



    print('-----------------------------------------------------------')
    audio_file_name = ROOT_PATH + "/audio_recording_" + args.user + ".wav"

    _file = open(ROOT_PATH + "/transcript_" + args.user + ".txt", "w")
    transcript = "\n".join(transcript.split(" "))
    _file.write(transcript + "\n")
    _file.close()

    try:
        print("Performing forced alignment..")
        msg = subprocess.check_output(["python", "-m", "aeneas.tools.execute_task", 
                                    audio_file_name, 
                                    ROOT_PATH + "/transcript_" + args.user + ".txt",
                                    #"task_language=eng|os_task_file_format=json|is_text_type=mplain",
                                    "task_language=eng|os_task_file_format=json|is_text_type=plain|task_adjust_boundary_nonspeech_min=0.0100|task_adjust_boundary_nonspeech_string=(sil)|task_adjust_boundary_algorithm=auto",
                                    ROOT_PATH + "/data_" + args.user + ".json", 
                                    "--presets-word"] )
    except subprocess.CalledProcessError as e:
        msg = e.output.decode("utf-8")
        print(msg)

    with open(ROOT_PATH + "/data_" + args.user + ".json") as f:
        data = json.load(f)
    
    """
    [
            {
                "word":"in",
                "start_time ":0.0,
                "duration":1.06
            },
            {
                "word":"clustering",
                "start_time ":1.06,
                "duration":0.52
            }]
    """

    words_list = []
    for word in data.get("fragments"):
        word_item = dict()

        if word["lines"][0] == "(sil)":
            continue

        word_item["word"] = word["lines"][0]
        word_item["start_time"] = float(word["begin"])
        word_item["duration"] = float(word["end"]) - float(word["begin"])
        words_list.append(word_item)

    print('-----------------------------------------------------------')

    return words_list