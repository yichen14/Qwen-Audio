pip install evaluate sacrebleu==1.5.1
pip install pycocoevalcap
pip install edit_distance editdistance
mkdir -p eval_audio/caption_evaluation_tools
git clone https://github.com/audio-captioning/caption-evaluation-tools.git eval_audio/caption_evaluation_tools
cd eval_audio/caption_evaluation_toolscoco_caption/
./get_stanford_models.sh
cd ../..
pip install sacrebleu
pip install sacrebleu\[ja\]
pip install sed_eval
pip install dcase_util