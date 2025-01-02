# LangChain と LangGraph による RAG・AI エージェント［実践］入門

本家リポジトリは[こちら](https://github.com/GenerativeAgents/agent-book)

## Environment

### dockerコンテナ ビルド & 起動

GPU環境が必須となっています

```bash
docker compose up -d --build
```

### コンテナにアタッチ

次にVScode左下の`><`ボタンより`コンテナで再度開く`でコンテナにアクセス

### 拡張機能インストール

無事コンテナが開いたら, 「拡張機能の推奨事項があります」という通知が出ると思います.
この通知を許可すると, `.vscode/extensions.json`に記載されている拡張機能が自動的にインストールされます.
もし通知が出なかった場合は, 左のメニューから`拡張機能`を選択し, `フィルターアイコン`->`推奨`‐>`インストールアイコン`を押せば一括インストールできます.
