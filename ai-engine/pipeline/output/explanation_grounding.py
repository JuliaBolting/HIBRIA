"""Rejeita inversões explícitas entre pontos e seus resultados.

É uma barreira conservadora de consistência, não uma prova semântica universal.
Associações lexicais contraditórias usam o texto fundamentado do fallback.
Paráfrases arbitrárias ainda exigem avaliação humana da qualidade.
"""
import re
from .explanation_context import norm, item_group

STOP = set("a o as os de do da dos das e em para com pelo pela por um uma que foi sao esta estao teve tem nao sobre entre apos segundo noticia informacao informacoes analise resultado verificacoes materiais encontrados recebeu confirmacao confirmou confirmada confirmado apoio suficiente sustentar sustentacao ponto parte conteudo abertura construcao".split())


def terms(text):
    return {word for word in re.findall(r"[a-z][a-z0-9-]{3,}", norm(text)) if word not in STOP}


def grounding_errors(text, context):
    points = [(terms(item.get("texto_selecionado", "")), item_group(item)) for item in context.get("itens", [])]
    issues = []
    for clause in re.split(r"[.!?;]|\b(?:mas|enquanto|porem)\b", norm(text)):
        limited = bool(re.search(r"(?:nao|sem).{0,45}(?:confirm|apoio|sustent|esclarec)|ficou em aberto|permaneceu em aberto", clause))
        supported = bool(re.search(r"(?:recebeu|receberam|teve|encontrou|encontraram|deu|deram).{0,25}(?:confirmacao|apoio)|(?:foi|foram|esta|estao).{0,15}confirmad|confirmou|confirmaram", clause))
        different = bool(re.search(r"diverg|contrad|nao coincid|apresent.{0,15}diferenc", clause))
        if not (limited or supported or different):
            continue
        if re.search(r"reputacao|veiculo|padroes de escrita|fonte", clause) and not (terms(clause) & set().union(*(t for t,g in points))):
            continue
        words = terms(clause)
        matches = [(len(words & tokens), group) for tokens, group in points if words & tokens]
        if supported and not limited and not any(group in {"support", "mixed"} for _, group in points):
            issues.append("apoio_declarado_sem_ponto_elegivel")
        if not matches:
            # Declarações gerais sobre o conjunto continuam sujeitas às regras
            # de formato/classificação; não viram comprovação de ponto concreto.
            continue
        best = max(score for score, _ in matches)
        groups = {group for score, group in matches if score == best}
        expected = {"open", "mixed", "difference"} if limited else {"mixed", "difference"} if different else {"support"}
        if not groups <= expected:
            issues.append("conclusao_diverge_do_ponto_avaliado")
    return sorted(set(issues))
