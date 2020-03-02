"""Server-end for the ASR demo."""
import os
import json
import subprocess
import time
import random
import argparse
import functools
from time import gmtime, strftime
import struct
import wave
import paddle.fluid as fluid
import numpy as np
import _init_paths
from data_utils.data import DataGenerator
from model_utils.model import DeepSpeech2Model
from data_utils.utility import read_manifest
from utils.utility import add_arguments, print_arguments


host_ip = "localhost"
host_port = 8086
beam_size = 500
num_conv_layers = 2
num_rnn_layers = 3
rnn_layer_size = 1024
alpha = 1.15
beta = 0.15
cutoff_prob = 1.0
cutoff_top_n = 40
use_gru = True
use_gpu = False
share_rnn_weights = False
speech_save_dir = "demo_cache"
warmup_manifest = "/home/Nishchith/DeepSpeech/data/tiny/manifest.test-clean"
mean_std_path = "/home/Nishchith/DeepSpeech/models/baidu_en8k/mean_std.npz"
vocab_path = "/home/Nishchith/DeepSpeech/models/baidu_en8k/vocab.txt"
model_path = "/home/Nishchith/DeepSpeech/models/baidu_en8k"
lang_model_path = "/home/Nishchith/DeepSpeech/models/lm/common_crawl_00.prune01111.trie.klm"
# decoding_method = "ctc_beam_search"
decoding_method = "ctc_greedy"
specgram_type = "linear"

def warm_up_test(audio_process_handler,
                 manifest_path,
                 num_test_cases,
                 random_seed=0):
    """Warming-up test."""
    manifest = read_manifest(manifest_path)
    rng = random.Random(random_seed)
    samples = rng.sample(manifest, num_test_cases)
    for idx, sample in enumerate(samples):
        print("Warm-up Test Case %d: %s", idx, sample['audio_filepath'])
        start_time = time.time()
        transcript = audio_process_handler(sample['audio_filepath'])
        finish_time = time.time()
        print("Response Time: %f, Transcript: %s" %
              (finish_time - start_time, transcript))


def start_server(args):
    """Start the ASR server"""
    # prepare data generator
    if use_gpu:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()

    data_generator = DataGenerator(
        vocab_filepath=vocab_path,
        mean_std_filepath=mean_std_path,
        augmentation_config='{}',
        specgram_type=specgram_type,
        keep_transcription_text=True,
        place = place,
        is_training = False)
    # prepare ASR model
    ds2_model = DeepSpeech2Model(
        vocab_size=data_generator.vocab_size,
        num_conv_layers=num_conv_layers,
        num_rnn_layers=num_rnn_layers,
        rnn_layer_size=rnn_layer_size,
        use_gru=use_gru,
        init_from_pretrained_model=model_path,
        place=place,
        share_rnn_weights=share_rnn_weights)

    vocab_list = [chars.encode("utf-8") for chars in data_generator.vocab_list]

    if decoding_method == "ctc_beam_search":
        ds2_model.init_ext_scorer(alpha, beta, lang_model_path,
                                  vocab_list)
    # prepare ASR inference handler
    def file_to_transcript(filename):
        feature = data_generator.process_utterance(filename, "")
        audio_len = feature[0].shape[1]
        mask_shape0 = (feature[0].shape[0] - 1) // 2 + 1
        mask_shape1 = (feature[0].shape[1] - 1) // 3 + 1
        mask_max_len = (audio_len - 1) // 3 + 1
        mask_ones = np.ones((mask_shape0, mask_shape1))
        mask_zeros = np.zeros((mask_shape0, mask_max_len - mask_shape1))
        mask = np.repeat(
            np.reshape(
                np.concatenate((mask_ones, mask_zeros), axis=1),
                (1, mask_shape0, mask_max_len)),
            32,
            axis=0)
        feature = (np.array([feature[0]]).astype('float32'),
                   None,
                   np.array([audio_len]).astype('int64').reshape([-1,1]),
                   np.array([mask]).astype('float32'))
        probs_split = ds2_model.infer_batch_probs(
            infer_data=feature,
            feeding_dict=data_generator.feeding)

        tik = time.time()
        if decoding_method == "ctc_greedy":
            result_transcript = ds2_model.decode_batch_greedy(
                probs_split=probs_split,
                vocab_list=vocab_list)
        else:
            result_transcript = ds2_model.decode_batch_beam_search(
                probs_split=probs_split,
                beam_alpha=alpha,
                beam_beta=beta,
                beam_size=beam_size,
                cutoff_prob=cutoff_prob,
                cutoff_top_n=cutoff_top_n,
                vocab_list=vocab_list,
                num_processes=1)
        
        print(time.time() - tik )
        return result_transcript[0]

    # warming up with utterrances sampled from Librispeech
    print('-----------------------------------------------------------')
    print('Warming up ...')
    audio_file_name = "/home/Nishchith/audio_recording_" + args.user + ".wav"
    # audio_file_name = "/home/Nishchith/2722020-163399.wav"
    transcript = file_to_transcript(audio_file_name)
    # transcript = file_to_transcript("/home/Nishchith/audio_samples/test-pravar_2.wav")
    
    _file = open("/home/Nishchith/transcript_" + args.user + ".txt", "w")
    transcript = "\n".join(transcript.split(" "))
    _file.write(transcript + "\n")
    _file.close()
    
    try:
        msg = subprocess.check_output(["python", "-m", "aeneas.tools.execute_task", 
                                    audio_file_name, 
                                    "/home/Nishchith/transcript_" + args.user + ".txt",
                                    #"task_language=eng|os_task_file_format=json|is_text_type=mplain",
                                    "task_language=eng|os_task_file_format=json|is_text_type=plain|task_adjust_boundary_nonspeech_min=0.0100|task_adjust_boundary_nonspeech_string=(sil)|task_adjust_boundary_algorithm=auto",
                                    "/home/Nishchith/data_" + args.user + ".json", 
                                    "--presets-word"] )
    except subprocess.CalledProcessError as e:
        msg = e.output.decode("utf-8")
        print(msg)

    with open("/home/Nishchith/data_" + args.user + ".json") as f:
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

    with open("/home/Nishchith/format_data_" + args.user + ".json", 'w') as f:
        json.dump(words_list, f)
    print('-----------------------------------------------------------')


def main():
    parser = argparse.ArgumentParser(description="USER ID")
    parser.add_argument(
        "--user",
        default="deepspeech_server",
        type=str,
        help="Model path",
    )

    args = parser.parse_args()
    start_server(args)


if __name__ == "__main__":
    main()
