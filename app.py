from flask import Flask, request, jsonify
from ctransformers import AutoModelForCausalLM

llm = AutoModelForCausalLM.from_pretrained(
    "zoltanctoth/orca_mini_3B-GGUF", model_file="orca-mini-3b.q4_0.gguf"
)

# Dicionário para armazenar o histórico de conversas, limitado a 20 respostas por thread_id
conversation_history = {}

def get_prompt(system: str, instruction: str, thread_id: str) -> str:
    history = conversation_history.get(thread_id, [])
    history_text = " ".join(history)
    prompt = f"### System:\n{system}\n\n### User:\n"
    if history:
        prompt += f"This is the conversation history: {history_text}. Now answer the question: "
    prompt += f"{instruction}\n\n### Response:\n"
    print(f"Prompt created: {prompt}")
    return prompt

app = Flask(__name__)

@app.route('/')
def home():
    return "Bem-vindo à API de demonstração!"

@app.route('/deebff5f-4fbf-4f06-bcb6-f54a2771c96f/chat', methods=['POST'])
def process_json():
    data = request.get_json()  # Recebe o JSON do corpo da requisição
    
    # Extrai dados do payload JSON
    system = data.get('system', '')
    instruction = data.get('instruction', '')
    thread_id = data.get('thread_id', '')  # Parâmetro thread_id obrigatório

    if not thread_id:
        return jsonify({"error": "thread_id is required"}), 400

    # Gera o prompt e obtém a resposta do modelo
    prompt = get_prompt(system, instruction, thread_id)
    response = llm(prompt)

    # Atualiza o histórico da conversa para este thread_id
    if thread_id in conversation_history:
        if len(conversation_history[thread_id]) >= 20:
            conversation_history[thread_id].pop(0)  # Remove a resposta mais antiga se o limite for alcançado
        conversation_history[thread_id].append(response)
    else:
        conversation_history[thread_id] = [response]

    # Retorna a resposta junto com o thread_id
    return jsonify({"response": response, "thread_id": thread_id})

if __name__ == '__main__':
    app.run(debug=True, port=80, host='0.0.0.0')