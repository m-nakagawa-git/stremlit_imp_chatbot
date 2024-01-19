import streamlit as st
import faiss
import numpy as np
import os
import openai
import pandas as pd
import time
import datetime  # @@

# # $$$$$$$$$$$$$$$$ langchain 使用時のコード start $$$$$$$$$$ !!!! $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# from langchain_openai import ChatOpenAI                          # api direct call 使用時はコメントアウト
# from langchain.chains import LLMChain                            # api direct call 使用時はコメントアウト
# from langchain.prompts import PromptTemplate                     # api direct call 使用時はコメントアウト
# # $$$$$$$$$$$$$$$$ langchain 使用時のコード start $$$$$$$$$$ !!!! $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# ページ設定
# print('==== st.set_page_config start ====', datetime.datetime.now())  # @@@
# start_time_1 = time.time()  # @@@
st.set_page_config(
    page_title="iMPRESS-AI Chatbot",
    page_icon="icon/AIイラスト.png",
)
# elapsed_time_1 = time.time() - start_time_1  # @@@
# print(f"==== st.set_page_config end ==== 処理にかかった時間: {elapsed_time_1}秒")  # @@@

# 2つの列を作成（左側の列にはタイトル、右側の列には画像を配置）
col1, col2 = st.columns([3, 1])  # 列の幅比を調整できます

# 左側の列にカスタムCSS タイトルを配置
# print('==== WINNDOW SET START ====', datetime.datetime.now())  # @@@
# start_time_2 = time.time()  # @@@
col1.markdown("""
    <style>
    .title-style {
        font-size: 35px;
        color: #FFFFFF;
    }
    </style>
    <div class="title-style">
        iMPRESS-AI Chatbot
    </div>
""", unsafe_allow_html=True)

# カスタムCSS
st.markdown("""
    <style>
    /* テキストエリアのラベルの色を変更するCSS */
    .stTextArea label {
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# カスタムCSSを定義
st.markdown("""
    <style>
    /* Streamlitボタンウィジェットのテキスト色を変更 */
    .stButton > button {
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# 右側の列に画像を配置
# 画像の解像度に合わせて適切なサイズを指定
# use_column_width=True は画像を列の幅に合わせて表示
# print('==== right roboimage_sub.png ====', datetime.datetime.now())  # @@@
col2.image("icon/roboimage_sub.png", use_column_width=True)


# 環境変数の読み込み
# from dotenv import load_dotenv
# load_dotenv()

st.sidebar.markdown("**＜問合せ内容選択＞**")  # 太字

# 言語選択のセレクトボックス
language = st.sidebar.selectbox(
    "言語を選択してください",
    ["選択してください",  # デフォルトの選択肢
     "日本語 (Japanese)",
     "英語 (English)"]
)
# セッション状態に言語を保存
st.session_state.language = language


# <<<<< 言語が選択されているかどうかをチェックする関数 >>>>>
def check_language_selection(language):
    """
    選択されていなければエラーメッセージを表示する。
    Parameters:
    language (str): セレクトボックスで選択された問い合わせ内容
    Returns:
    bool: 言語が適切に選択されている場合はTrue、そうでなければFalse
    """
    if language == "選択してください":  # デフォルト選択の状態
        st.error("言語を選択してください。")
        return False
    return True


# 問合わせ内容のセレクトボックス（デフォルトオプション追加）
inquiry = st.sidebar.selectbox(
    "問合わせ内容を選択してください",
    ["選択してください",  # デフォルトの選択肢
     "iFOCUS Next",
     "iFUSION Client",
     "iFUSION Designer",
     "iFUSION Portal",
     "その他のお問い合わせ"]      # ←はじめてのiFUSIONについて含む
)
# セッション状態に問合わせ内容を保存
st.session_state.inquiry = inquiry
# elapsed_time_2 = time.time() - start_time_2  # @@@


# <<<<< 問い合わせ内容が選択されているかどうかをチェックする関数 >>>>>
def check_inquiry_selection(inquiry):
    """
    選択されていなければエラーメッセージを表示する。
    Parameters:
    inquiry (str): セレクトボックスで選択された問い合わせ内容
    Returns:
    bool: 問い合わせ内容が適切に選択されている場合はTrue、そうでなければFalse
    """
    if inquiry == "選択してください":   # デフォルト選択の状態
        st.error("問い合わせ内容を選択してください。")
        return False
    return True


# サイドバーの設定
for _ in range(2):
    st.sidebar.write("")
st.sidebar.image("icon/roboimage_last.png", width=400)

# print(f"==== WINNDOW SET ==== 処理にかかった時間: {elapsed_time_2}秒")  # @@@
# print('==== WINNDOW SET END  ====', datetime.datetime.now())  # @@@


# セッション状態に初期起動フラグを設定
# （初期起動時はTrueにする）
if 'is_first_run' not in st.session_state:
    st.session_state['is_first_run'] = True
    # print('==== No1 : is_first_run = True  ====', st.session_state['is_first_run'])  # @@

# 初期起動時にのみ行う処理
if st.session_state['is_first_run']:

    # 送信ボタン押下回数の初期化
    st.session_state.send_btn_count = 0
    # print('==== No2 : is_first_run  ====', st.session_state['is_first_run'])  # @@
    # print('==== No2 : st.session_state.send_btn_count && ====', st.session_state.send_btn_count)  # @@

    # 初期起動フラグをFalseに設定
    st.session_state['is_first_run'] = False
    # print('==== No3 : is_first_run = False  ====', st.session_state['is_first_run'])  # @@

# セッション状態の初期化（初回起動時のみ）
if "generated" not in st.session_state:
    st.session_state.generated = []
if "past" not in st.session_state:
    st.session_state.past = []


# 送信ボタンがクリックの確認
# （後続処理を送信ボタン押下時のみ行う時の判断に使用）
if "button_clicked" not in st.session_state:
    st.session_state.button_clicked = False
    # print('##### No5 : send_button False set #####')

# # $$$$$$$$$$$$$$$$ langchain 使用時のコード start $$$$$$$$$$ !!!! $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# template = (
#     "あなたは株式会社インプレスのパッケージ商品の優秀なカスタマーサポート担当です。\n"
#     "以下の制約条件に従って、株式会社インプレスのお問い合わせ窓口チャット\n"
#     "ボットとしてユーザーからの最新質問に回答して下さい。\n"
#     "---\n"
#     "# 制約条件:\n"
#     "- ユーザーからの最新質問に対して、回答時の参考情報を参考にして回答文を生成して下さい。\n"
#     "- 回答内容は、ユーザーからの最新質問に対してのみ回答して下さい。\n"
#     "- 回答できないと判断した場合は、回答文は「回答不能」とセットして下さい。\n"
#     "- 回答内容には、回答時の参考情報の中に含まれているユーザーからの質問には回答しないで下さい。\n"
#     "- 回答は見出し、箇条書き、表などを使って人間が読みやすく表現してください。\n"
#     "\n"
#     "# ユーザーからの最新質問:\n"
#     "{question}\n"
#     "\n"
#     "#回答時の参考情報:\n"
#     "{history}\n"
#     "\n"
#     "# 回答文:\n"
#     )

# # 会話プロンプトを作成
# prompt = PromptTemplate(
#     input_variables=["question", "history"],
#     template=template
# )

# # ----------------------------------------------------------------------------
# # <<<<< 会話の読み込みを行う関数 >>>>>
# #  gpt-4-1106-preview  gpt-3.5-turbo-16k
# @st.cache_resource
# def load_conversation():
#     print('==== def load_conversation start ====')
#     start_time = time.time()  # @@@
#     llm = ChatOpenAI(
#         model_name="gpt-3.5-turbo-16k",
#         temperature=0
#     )
#     conversation = LLMChain(
#         llm=llm,
#         prompt=prompt,
#         verbose=True)
#     elapsed_time = time.time() - start_time  # @@@
#     print(f"==== def load_conversation ==== 処理にかかった時間: {elapsed_time}秒")  # @@@
#     print('==== def load_conversation end ====')
#     return conversation

# # $$$$$$$$$$$$$$$$ langchain 使用時のコード end $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# &&&&&&&&&&&&&&&& api direct call 使用時のコード start &&&&&&&&&& !!!! &&&&&&&&&&&&&&&&&&&&&&&&&
# Chatgpt api call 
# gpt-4-1106-preview  gpt-3.5-turbo-16k
@st.cache_resource
def createCompletion(prompt):
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            messages=prompt
        )
        return response.choices[0].message.content
    except Exception as e:
        print(e)
        raise e


def load_conversation(user_message, conversation_history):
    history  = conversation_history
    system_msg = f"""
    あなたは株式会社インプレスのパッケージ商品の優秀なカスタマーサポート担当です。
    以下の制約条件に従って、株式会社インプレスのお問い合わせ窓口チャット
    ボットとしてユーザーからの最新質問に回答して下さい。
    ---
    # 制約条件:
    - ユーザーからの質問に対して、回答時の参考情報を参考にして回答文を生成して下さい。
    - 回答内容は、ユーザーからの最新質問に対してのみ回答して下さい。
    - 回答できないと判断した場合は、回答文は「回答不能」とセットして下さい。
    - 回答内容には、回答時の参考情報の中に含まれているユーザーからの質問には回答しないで下さい。
    - 回答は見出し、箇条書き、表などを使って人間が読みやすく表現してください。

    #回答時の参考情報:
    {history}

    # 回答文:
    """

    prompt = [
         {"role": "system", "content": system_msg},
         {"role": "user", "content": user_message}]
    
    # プロンプトを生成し、Completion APIを使用して回答を生成します
    # print('%%%% pronpt %%%%',prompt)
    completion = createCompletion(prompt)
    return completion
# &&&&&&&&&&&&&&&& api direct call 使用時のコード end &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&


# ----------------------------------------------------------------------------
# <<<<< 送信ボタンを押したときの処理を行う関数 >>>>>
def on_input_change():
    # print('==== def on_input_change start ====',
    #       datetime.datetime.now())  # @@@
    # start_time_a = time.time()  # @@@

    # セッション状態から language の値を取得
    language = st.session_state.language

    # セッション状態から inquiry の値を取得
    inquiry = st.session_state.inquiry

    # 送信ボタン押下回数
    st.session_state.send_btn_count += 1
    # print('----------- No6 session_state.send_btn_count UP  --------------',
    #       st.session_state.send_btn_count)

    # 最初の送信ボタン押下時
    if st.session_state.send_btn_count == 1:

        # 言語の選択チェックの関数を呼び出し
        if not check_language_selection(language):  # 未選択時エラー
            st.session_state.send_btn_count = 0     # 送信
            return  # 言語を未選択の場合、ここで処理を終了

        # 問い合わせ内容の選択チェックの関数を呼び出し
        if not check_inquiry_selection(inquiry):
            st.session_state.send_btn_count = 0
            return  # 問い合わせ内容を未選択の場合、ここで処理を終了

        # print('##### Parquetファイル読み込み開始 #####')  # @@
        # start_time_b = time.time()

        # 問い合わせ内容に応じてParquetファイルを選択
        parquet_files = {
            "iFOCUS Next": "output_data_Next.parquet",
            "iFUSION Client": "output_data_Client.parquet",
            "iFUSION Designer": "output_data_Designer.parquet",
            "iFUSION Portal": "output_data_Portal.parquet",
            "その他のお問い合わせ": "output_data_others.parquet"
        }

        # Parquetファイルからデータを読み込む
        parquet_fname = parquet_files.get(inquiry)

        # Parquetファイルからデータを読み込む
        parquet_df = load_parquet_file(parquet_fname)
        
        # 読み込んだデータフレームをセッション状態に保存
        st.session_state['parquet_df'] = parquet_df
        # print('@@@@@ parquet_df%% @@@@@', parquet_df)  # @@F
        # elapsed_time_b = time.time() - start_time_b  # @@@
        # print(f"==== Parquetファイル読み込み ==== 処理にかかった時間: {elapsed_time_b}秒")  # @@@
        # print('##### Parquetファイル読み込み修了 #####')  # @@

    # この関数に入る場合は送信ボタンを押下しているので
    # Trueにする
    st.session_state.button_clicked = True
    # print('##### No9 : send_button True set #####')

    # 今回のUSERMSGをベクトル化してFaissデータと類似性検索を行う
    user_message = st.session_state.user_message
    similar_text_indices = get_similar_texts(user_message, inquiry)
    # print('@@@@@ similar_text_indices @@@@@', similar_text_indices)

    # 回答に関連する画像格納用エリアをクリア
    image_paths_to_display = []
 
    # ユーザーとボットの過去の会話履歴を組み合わせる
    conversation_history = ""
    for i, (user_msg, bot_msg) in enumerate(
        zip(st.session_state.past, st.session_state.generated), start=1):
        conversation_history += (f"(history{i}) User: {user_msg}\n"
                                f" Bot: {bot_msg['text']}\n")
    
    # print('@@@@@ conversation_history 11 @@@@@', conversation_history)
    
    # 各回の会話毎に画像表示用のセッションステートをリセット
    st.session_state['display_image_path'] = None

    # USERMSGと類似性検索でヒットしたデータのindex-noから
    # 該当のtextデータと画像データ格納場所のPATHを取得する
    if similar_text_indices:
        for index in similar_text_indices:
            # print('@@@@@ index %% @@@@@', index)
            text, image_paths = retrieve_text(index)
            # 会話履歴格納エリアに該当のtextデータを追加する
            conversation_history += " " + text
            # print('@@@@@ conversation_history 22 @@@@@', conversation_history)
            # 画像格納用エリアに該当の画像データ格納場所のPATHを追加する
            image_paths_to_display.extend(image_paths)

    # langchain.chainsでChatGPTから回答を得る
    # print('@@@@@ conversation_history 33 @@@@@', conversation_history)
    # 
    # start_time_8 = time.time()  # @@@
    # # $$$$$$$$$$$$$$$$ langchain 使用時のコード start $$$$$$$$$$ !!!! $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    # question = user_message 
    # history  = conversation_history
    # conversation = load_conversation() 
    # answer_text = conversation.predict(
    #     question=question, history=history)
    # # $$$$$$$$$$$$$$$$ langchain 使用時のコード end  $$$$$$$$$$ !!!! $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
    # &&&&&&&&&&&&&&&& api direct call 使用時のコード start &&&&&&&&&& !!!! &&&&&&&&&&&&&&&&&&&&&&&&&
    answer_text = load_conversation(user_message, conversation_history) 
    # &&&&&&&&&&&&&&&& api direct call 使用時のコード end  &&&&&&&&&& !!!! &&&&&&&&&&&&&&&&&&&&&&&&&
    
    # APIからの応答をチェック
    # print(' 回答不能 確認 ',answer_text)
    if "回答不能" in answer_text:
        answer_text = f'''
        申し訳ありませんが、ご質問頂いた内容についての情報を持ち合わせていませんので
        ご回答できません。必要であれば弊社のサポートチームへ連絡をお願い致します。
        (Tel:xxxxxx, Email:xxxxx@imprex.co.jp)
        '''

    # elapsed_time_8 = time.time() - start_time_8  # @@@
    
    # print(f"●●●●●●●●●●●● conversation.predict ●●●●●●●●●●●●● 処理にかかった時間: {elapsed_time_8}秒")  # @@@
    # print('answer_text',answer_text)
    
    # チャットボット側の履歴に追加する
    st.session_state.generated.append(
        {"text": answer_text, "images": image_paths_to_display})

    # USER側の履歴に追加する
    st.session_state.past.append(user_message)
    
    # send_btn_countが11になった場合、最も古い履歴を削除
    if st.session_state.send_btn_count >= 3:  # change MAX件数+1を指定する
        st.session_state.generated.pop(0)     # 最も古い要素を削除
        st.session_state.past.pop(0)          # 最も古い要素を削除

    # 今回のUSERMSGをクリア
    st.session_state.user_message = ""

    # elapsed_time_a = time.time() - start_time_a  # @@@
    # print(f"●●●●●●●●●●●● def on_input_change ●●●●●●●●●●●●● 処理にかかった時間: {elapsed_time_a}秒")  # @@@
    # print('==== def on_input_change end ====',
    #       datetime.datetime.now())  # @@@


# ----------------------------------------------------------------------------
# <<<<< Faissを用いた類似性検索を行う関数 >>>>>
def get_similar_texts(
        user_input, inquiry,
        top_k=10, hit_rate_threshold=0.50
):
    # print('==== def get_similar_texts start ====',
    #        datetime.datetime.now())  # @@@
    # start_time_d = time.time()  # @@@

    # 最初の送信ボタン押下時
    if st.session_state.send_btn_count == 1:
        # 問い合わせ内容に応じてFaissインデックスファイルを選択
        index_files = {
            "iFOCUS Next": "faiss_index_Next.idx",
            "iFUSION Client": "faiss_index_Client.idx",
            "iFUSION Designer": "faiss_index_Designer.idx",
            "iFUSION Portal": "faiss_index_Portal.idx",
            "その他のお問い合わせ": "faiss_index_others.idx"
        }
        # 問合せ内容に対応するファイル名を取得
        index_path = index_files[inquiry]
        # print('@@@@@ index_path @@@@@', index_path)
        # 読み込んだindexをセッション状態に保存
        # print('##### Faissデータ読み込み開始 #####')  # @@
        # start_time_b = time.time()
        st.session_state['index'] = load_faiss_index(index_path)
        # print('@@@@@ index @@@@@', st.session_state['index'])
        # elapsed_time_b = time.time() - start_time_b  # @@@
        # print(f"●●●●●●●●●●●● Faissデータ読み込み ●●●●●●●●●●●●● 処理にかかった時間: {elapsed_time_b}秒")  # @@@


    # USER入力のベクトル化    
    # start_time_e = time.time()  # @@@
    user_embedding = get_embedding(user_input)
    # elapsed_time_e = time.time() - start_time_e  # @@@
    # print(f"●●●●●●●●●●●● get_embedding  ●●●●●●●●●●●●● 処理にかかった時間: {elapsed_time_e}秒")  # @@@

    # start_time_n = time.time()  # @@@
    user_embedding_normalized = normalize_L2(
        np.array(user_embedding).reshape(1, -1))
    # elapsed_time_n = time.time() - start_time_n  # @@@
    # print(f"●●●●●●●●●●●● normalize_L2  ●●●●●●●●●●●●● 処理にかかった時間: {elapsed_time_n}秒")  # @@@

    # 類似性検索を実行
    # start_time_f = time.time()  # @@@
    index = st.session_state['index']
    similarity_scores, similar_indices = index.search(
        user_embedding_normalized, top_k)
    # elapsed_time_f = time.time() - start_time_f  # @@@
    # print(f"●●●●●●●●●●●● 類似性検索実行  ●●●●●●●●●●●●● 処理にかかった時間: {elapsed_time_f}秒")  # @@@

    # ヒット率の閾値に基づいて最もスコアが高いテキストのインデックスを返す
    max_score_idx = None
    max_score = hit_rate_threshold
    for score, idx in zip(similarity_scores[0], similar_indices[0]):
        if score > max_score:
            max_score = score
            max_score_idx = idx

    # 最高スコアのテキストのインデックスを返す（存在しない場合はNone）
    # elapsed_time_d = time.time() - start_time_d  # @@@
    # print(f"●●●●●●●●●●●● def get_similar_texts  ●●●●●●●●●●●●● 処理にかかった時間: {elapsed_time_f}秒")  # @@@
    # print('##### def get_similar_texts RETURN #####',
        #    datetime.datetime.now())  # @@
    return [max_score_idx] if max_score_idx != -1 else []


# ----------------------------------------------------------------------------
# <<<<< USER入力のベクトル化を行う関数 parquet からデータ取得 >>>>>
def get_embedding(text, model="text-embedding-ada-002"):
    # print('##### def get_embedding start #####',
    #       datetime.datetime.now())  # @@
    text = text.replace("\n", " ")
    # print('##### def get_embedding end #####', datetime.datetime.now())  # @@
    return openai.embeddings.create(
        input=[text], model=model).data[0].embedding


# ----------------------------------------------------------------------------
# <<<<< ベクトルの正規化を行う関数 >>>>>
def normalize_L2(x):
    return x / np.linalg.norm(x, ord=2, axis=1, keepdims=True)


# ----------------------------------------------------------------------------
# <<<<< parquet からデータ取得 >>>>>
# 指定されたインデックスに対応するテキストと画像パスを
# 取得してリターンする
def retrieve_text(index):
    # print('##### def retrieve_text START #####', datetime.datetime.now())  # @@
    # セッション状態から parquet_df を取得
    # start_time_g = time.time()  # @@@
    parquet_df = st.session_state['parquet_df']
    # print('parquet_df', parquet_df)
    filtered_df = parquet_df[parquet_df['index-no'] == index]
    if not filtered_df.empty:
        data = filtered_df.to_dict('records')[0]
        text = data['文章テキスト']
        image_paths = [
            os.path.join(data['画像path_bot'], data[f'画像{i}'])
            for i in range(1, data['参照画像数'] + 1)
        ]
        # elapsed_time_g = time.time() - start_time_g  # @@@
        # print(f"●●●●●●●●●●●● def retrieve_text  ●●●●●●●●●●●●● 処理にかかった時間: {elapsed_time_g}秒")  # @@@        
        # print('##### def retrieve_text end #####',
            #   datetime.datetime.now())  # @@
        return text, image_paths
    else:
        # print('##### def retrieve_text end #####',
        #       datetime.datetime.now())  # @@
        return "", []


# ----------------------------------------------------------------------------
# <<<<< 画像表示用の関数を定義 >>>>>
# USERから画像表示リクエストがあった場合に画像の下に
# ファイルネームを表示する
def display_image_as_link(image_path):
    # print('##### def display_image_as_link START #####',
    #       datetime.datetime.now())  # @@
    file_name = os.path.basename(image_path)
    st.markdown(f"[{file_name}](image_path)", unsafe_allow_html=True)
    # print('##### def display_image_as_link START #####',
    #       datetime.datetime.now())  # @@

# print('##### point06 #####', datetime.datetime.now())


# 質問入力欄と送信ボタンを設置
with st.container():
    # print('------------------------------------------------------ input wait -----------------------------------', datetime.datetime.now())
    user_message = st.text_area(
        "＜＜質問内容をテキスト入力後、Ctrl+Enterで入力を確定させてから、送信ボタンを押して下さい＞＞",
        key="user_message"    # 高さ3行分 最小値
    )
    st.button("送信", on_click=on_input_change)
    
    
# ----------------------------------------------------------------------------
# parquet データ取得
@st.cache_data
def load_parquet_file(parquet_fname):
    return pd.read_parquet(parquet_fname)


# ----------------------------------------------------------------------------
# Faiss データ取得
@st.cache_data
def load_faiss_index(index_path):
    return faiss.read_index(index_path)


# ----------------------------------------------------------------------------
# print('##### point05 #####', datetime.datetime.now()) # @@
# 会話履歴を表示するためのスペースを確保
chat_placeholder = st.empty()

# print('$$$$$$$$$$$$ 画面リセット終了 &&&&&&&&&&&&&', datetime.datetime.now())
# # 会話履歴の表示と参考画像の一覧表表示
if st.session_state.button_clicked:    
    # print('##### point08 #####')
    # start_time_9 = time.time()  # @@@
    img_count = 0
    with chat_placeholder.container():
        # 会話履歴を逆順にして新しいメッセージが最上位に表示されるようにする
        # print('##### point08 kaiwa rireki start #####',
        # datetime.datetime.now())
        for i in reversed(range(len(st.session_state.generated))):
        # for i in range(len(st.session_state.generated)):
            user_outmsg = st.session_state.past[i]
            bot_outmsg = st.session_state.generated[i]

            st.markdown("---")
            col1, col2 = st.columns([1, 15])
            with col1:
                st.image("icon/human.png", width=35)
            with col2:
                # st.markdown(f"<p style='color: white;'>{user_outmsg}</p>",
                #              unsafe_allow_html=True)
                # st.write(user_outmsg)
                user_outmsg_lines = user_outmsg.split('\n')
                for line in user_outmsg_lines:
                    st.markdown(f"<p style='color: white'>{line}</p>", unsafe_allow_html=True)
            
            st.markdown("---")
            col1, col2 = st.columns([1, 15])
            with col1:
                st.image("icon/bot.png", width=35)
            with col2:
                # HTMLスタイルを適用したテキストを生成
                # st.markdown(f"<p style='color: white'>{bot_outmsg['text']}</p>",
                #             unsafe_allow_html=True)
                # st.write(bot_outmsg["text"])
                bot_outmsg_lines = bot_outmsg['text'].split('\n')
                for line in bot_outmsg_lines:
                    st.markdown(f"<p style='color: white'>{line}</p>", unsafe_allow_html=True)

              

            # 画像表示用のセッションステートを初期化
            if 'display_image_path' not in st.session_state:
                st.session_state['display_image_path'] = None

            # 関連画像に対応するボタンを表示
            if bot_outmsg["images"]:
                # print('@@@ point-a @@@ ')
                for index, image_path in enumerate(bot_outmsg["images"]):
                    file_name = os.path.basename(image_path)
                    img_count += 1
                    # ボタンに一意のキーを割り当て
                    if st.button(file_name, key=f"image_button_{img_count}"):
                        st.session_state['display_image_path'] = image_path

            # 選択された画像を表示
            if st.session_state['display_image_path']:
                # 画像のファイル名を取得して表示
                file_name = os.path.basename(
                    st.session_state['display_image_path'])
                st.caption(file_name)
                st.image(
                    st.session_state['display_image_path'],
                    use_column_width=True)

    # elapsed_time_9 = time.time() - start_time_9  # @@@
    # print(f"●●●●●●●●●●●● 会話履歴の表示と参考画像の一覧表表示  ●●●●●●●●●●●●● 処理にかかった時間: {elapsed_time_9}秒")  # @@@    
