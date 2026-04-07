from transformers import pipeline
# Referenciando o modelo BERTimbau adaptado para sentimentos
# https://huggingface.co/pysentimiento/bertweet-pt-sentiment
classificador_pt = pipeline("sentiment-analysis", model="pysentimiento/bertweet-pt-sentiment")

# Teste
frases = [
    "Seu atendimento foi fantástico, estou muito satisfeita!",
    "Achei o novo filme 'cabra bom de bola' animal, valeu cada centavo",
    "O produto é cheio de defeitos e o suporte técnico não me ajudou em nada!"
]

resultados = classificador_pt(frases)
for i, res in enumerate(resultados):
    print(f"Texto: {frases[i]}")
    print(f"Resultado: {res['label']} | Confiança: {res['score']:.2%}\n")
