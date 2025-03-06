import os  # OSの機能にアクセスするためのモジュール（環境変数取得など）
import tornado.ioloop  # Tornadoの非同期イベントループを提供するモジュール
from tornado.options import define, options, parse_command_line  # コマンドライン引数の処理用
from application import Application  # 自作のTornadoアプリケーションクラス


# コマンドラインオプションの定義：ポート番号を指定可能にする（デフォルト8888）
define("port", default=8888, help="run on the given port", type=int)


def main():
    """
    メイン関数：Tornadoウェブサーバーを設定して起動する
    """
    parse_command_line()  # コマンドライン引数を解析
    app = Application()  # アプリケーションのインスタンスを作成
    
    # ポート番号を決定：環境変数があればそれを使用、なければデフォルト8888
    port = int(os.environ.get("PORT", 8888))
    
    app.listen(port)  # 指定されたポートでサーバーを起動
    print("Run server on port: {}".format(port))  # 起動情報を表示
    
    # イベントループを開始（サーバーの実行を継続）
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    # このスクリプトが直接実行された場合のみmain関数を実行
    main()
