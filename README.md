# Talia AI Daemon

Um daemon Python que utiliza o modelo Gaia (CEIA-UFG/Gemma-3-Gaia-PT-BR-4b-it) para resumir textos em português, expondo um servidor gRPC sobre Unix Domain Socket (UDS).

## Requisitos

- Python 3.8+
- pip (gerenciador de pacotes Python)

## Instalação

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/talia-ai-daemon.git
   cd talia-ai-daemon
   ```

2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```

## Configuração

O daemon utiliza variáveis de ambiente para configuração. Você pode definir o caminho do socket gRPC através de um arquivo `.env` ou `talia_daemon.env`. Por exemplo:

```
SOCKET_PATH=/tmp/daemon.sock
```

Se não for definido, o padrão é `/tmp/daemon.sock`.

## Uso

### Iniciar o Daemon

Para iniciar o daemon:
```bash
python talia_daemon.py
```

O daemon irá:
- Carregar o modelo Gaia
- Iniciar o servidor gRPC sobre UDS no caminho configurado
- Registrar logs em `talia_daemon.log`

### Testar o Daemon

Para testar o daemon, você pode usar o cliente gRPC fornecido. Certifique-se de que o daemon esteja rodando e execute:

```bash
python client.py
```

O cliente irá:
- Verificar a saúde do daemon
- Enviar uma solicitação de resumo e exibir o resultado

## Testes

Os testes são escritos usando pytest. Para executar os testes:

```bash
pytest test_talia_daemon.py
```

## Logs

Os logs são salvos em `talia_daemon.log` e também são exibidos no console.

## Personalização

Para modificar o comportamento do daemon, edite o arquivo `talia_daemon.py`:
- Altere o intervalo de processamento modificando o valor em `time.sleep()`
- Modifique o texto de exemplo na função `run()`
- Ajuste o nível de logging conforme necessário 