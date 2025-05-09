from flask import Flask, request, jsonify
from ckiptagger import WS, POS, NER, data_utils
from opencc import OpenCC
import os

MODEL_DIR = "./data"

def ensure_ckip_model():
    if not os.path.exists(MODEL_DIR):
        print("下載 CkipTagger 模型中...")
        data_utils.download_data_gdown(MODEL_DIR)
        print("模型下載完成")

ensure_ckip_model()

# 初始化模型
ws = WS(MODEL_DIR)
pos = POS(MODEL_DIR)
ner = NER(MODEL_DIR)
cc = OpenCC('t2s')

app = Flask(__name__)

@app.route("/segment", methods=["POST"])
def segment():
    text = request.json.get("text", "")
    word_sentence_list = ws([text])
    return jsonify({"segment": word_sentence_list[0]})

@app.route("/pos", methods=["POST"])
def pos_analysis():
    text = request.json.get("text", "")
    word_sentence_list = ws([text])
    pos_sentence_list = pos(word_sentence_list)
    result = [(w, p) for w, p in zip(word_sentence_list[0], pos_sentence_list[0])]
    return jsonify({"pos": result})

@app.route("/ner", methods=["POST"])
def ner_analysis():
    text = request.json.get("text", "")
    word_sentence_list = ws([text])
    pos_sentence_list = pos(word_sentence_list)
    ner_sentence_list = ner(word_sentence_list, pos_sentence_list)
    return jsonify({"ner": ner_sentence_list[0]})

@app.route("/convert", methods=["POST"])
def convert():
    text = request.json.get("text", "")
    converted = cc.convert(text)
    return jsonify({"converted": converted})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
