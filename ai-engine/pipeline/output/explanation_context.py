"""Contrato da redação: dados observados, unidades explícitas e nenhuma busca.

O contexto completo permanece disponível no laboratório. A versão enviada ao
modelo é limitada de forma distribuída, com contagem explícita das omissões.
Não recalcula notas de análises antigas nem promove similaridade a confirmação.
"""
from __future__ import annotations

import copy
import html
import json
import math
import re
import unicodedata
from types import SimpleNamespace
from typing import Any

VERSION = "explanation-7.0.0"


def value(obj: Any, key: str, default=None):
    return obj.get(key, default) if isinstance(obj, dict) else getattr(obj, key, default)


def plain(text: Any) -> str:
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]*>", " ", str(text or "")))).strip()


def norm(text: Any) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", plain(text).lower())
                   if unicodedata.category(c) != "Mn")


def number(obj: Any):
    try:
        num = float(obj)
        return num if math.isfinite(num) else None
    except (TypeError, ValueError):
        return None


def excerpt(text: str, limit: int) -> str:
    text = plain(text)
    if len(text) <= limit:
        return text
    return text[:max(0, limit - 1)].rsplit(" ", 1)[0] + "…"


def _as_dict(obj):
    if isinstance(obj, dict):
        return copy.deepcopy(obj)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    return dict(vars(obj)) if hasattr(obj, "__dict__") else {}


def build_context(result) -> dict:
    breakdown = value(result, "score_breakdown", {}) or {}
    stances = [_as_dict(s) for s in (value(result, "stance_results", []) or [])]
    counts = {key: int((breakdown.get("stance_stats") or {}).get(key, 0) or 0)
              for key in ("support", "contradict", "neutral", "insufficient")}
    coverage = number(breakdown.get("coverage_score"))  # SEMPRE escala 0..100.
    label = plain(value(result, "label_final", "não verificado")) or "não verificado"
    items = []
    trust_counts = {"aprovada": 0, "nao_aprovada_ou_nao_avaliada": 0, "desconhecida": 0}
    contextual = 0
    failed_layers = set()
    for retrieval in value(result, "retrieval_results", []) or []:
        claim = value(retrieval, "claim", {}) or {}
        claim_id = str(value(claim, "claim_id", ""))
        relations = [s for s in stances if str(s.get("claim_id", "")) == claim_id]
        refs = []
        for ev in value(retrieval, "evidences", []) or []:
            metadata = value(ev, "metadata", {}) or {}
            evidence_id, url = value(ev, "evidence_id", ""), value(ev, "url", "")
            # Nunca associar apenas pelo domínio: um site pode ter vários textos.
            matched = [s for s in relations if
                       (evidence_id and s.get("evidence_id") == evidence_id) or
                       (url and s.get("url") == url)]
            relation = matched[0] if len(matched) == 1 else {}
            trusted = value(ev, "trusted_source")
            accepted_by_policy = (value(ev, "retrieval_layer") == "factcheck" or
                                  value(ev, "source_type") == "fact_check" or
                                  (value(ev, "retrieval_layer") == "vector_store" and
                                   metadata.get("approved_for_rag") is True))
            status = ("aprovada" if trusted is True or accepted_by_policy else
                      "nao_aprovada_ou_nao_avaliada" if trusted is False else "desconhecida")
            trust_counts[status] += 1
            is_context = (value(ev, "source_type") == "encyclopedia" or
                          value(ev, "retrieval_layer") == "wikipedia")
            contextual += int(is_context)
            refs.append({
                "id": evidence_id,
                "texto": plain(value(ev, "text", "")),
                "titulo": plain(value(ev, "title", "")),
                "fonte": plain(value(ev, "source", "")),
                "url": url,
                "confianca_na_origem": status,
                "somente_contexto": is_context,
                "similaridade_recuperacao": number(value(ev, "similarity")),
                "sinal_automatico": relation.get("stance") or value(ev, "stance"),
                "motivo_do_sinal": relation.get("reason") or metadata.get("stance_reason"),
                "similaridade_no_sinal": number(relation.get("similarity") or metadata.get("stance_similarity")),
            })
        layers = value(retrieval, "layers_failed", []) or []
        failed_layers.update(str(layer) for layer in layers)
        items.append({"id": claim_id, "texto_selecionado": plain(value(claim, "text", "")),
                      "referencias": refs, "registros_automaticos": relations})

    reputation = number(breakdown.get("reputation_score"))
    textual = number(breakdown.get("bertimbau_score"))
    weights = breakdown.get("weights") or {}
    if breakdown.get("reputation_status") not in (None, "evaluated") or weights.get("reputation") == 0:
        reputation = None
    if breakdown.get("bertimbau_status") not in (None, "ok") or weights.get("bertimbau") == 0:
        textual = None
    # Apenas os valores/status realmente salvos. Dados ausentes não viram zero.
    signals = {
        "origem": {"nota": reputation, "status": breakdown.get("reputation_status")},
        "texto": {"nota": textual, "status": breakdown.get("bertimbau_status")},
        "similaridade_media_dos_itens_aceitos": number(breakdown.get("evidence_score")),
        "peso_dos_componentes": breakdown.get("weights", {}),
        "contagens": counts,
        "registros_sem_referencia": sum(not s.get("evidence_id") and not s.get("url") for s in stances),
        "origens_das_referencias": trust_counts,
        "referencias_contextuais": contextual,
        "falhas_de_recuperacao": sorted(failed_layers),
    }
    result_code = norm(label)
    if result_code == "nao confiavel":
        reason = "sinais_de_contradicao" if counts["contradict"] else "motivo_nao_documentado"
    elif result_code == "confiavel":
        reason = "criterios_para_confiavel_atendidos"
    elif result_code == "parcialmente confiavel":
        reason = "criterios_atendidos_parcialmente"
    elif coverage == 0:
        reason = "nenhum_item_aceito_no_calculo"
    elif result_code == "evidencia insuficiente" and coverage is not None and coverage < 100 * (number(breakdown.get("min_coverage_for_partial")) if number(breakdown.get("min_coverage_for_partial")) is not None else .2):
        reason = "poucos_itens_aceitos_no_calculo"
    else:
        reason = "resultado_inconclusivo"

    return {
        "versao": VERSION,
        "decisao": {"rotulo": label, "indice": number(value(result, "score_final")),
                    "motivo": reason, "cobertura_percentual": coverage,
                    "unidade_cobertura": "itens selecionados com referencia aceita no calculo; nao e porcentagem de verdade nem do texto inteiro",
                    "rastro_do_calculo": breakdown.get("decision_trace")},
        "sinais": signals,
        "interpretacao": {
            "contagens": "support/contradict/neutral sao sinais por comparacao; insufficient tambem inclui itens sem referencia. Nao somar como numero de fatos ou de noticias.",
            "similaridade": "semelhanca entre textos nao comprova acontecimentos; sinais sao heuristicas, nao checagens humanas",
            "confianca": "origem nao aprovada pode estar sem avaliacao; isso nao a torna falsa ou desonesta",
            "auxiliares": "reputacao avalia o veiculo; classificador textual reconhece padroes de escrita. Nenhum comprova fatos sozinho",
        },
        "noticia": {"titulo": plain(value(result, "title", "")),
                    "texto": plain(value(result, "clean_text", "") or " ".join(value(result, "blocks_clean", []) or []) or value(result, "raw_text", "") or value(result, "content", ""))},
        "itens": items,
    }


def compact_context(context: dict, max_chars: int) -> dict:
    """Redução distribuída: não privilegia as primeiras afirmações da notícia.

    A contagem de caracteres não equivale à tokenização do Qwen. É um teto
    conservador configurável; métricas reais são registradas pelo gerador.
    """
    source = copy.deepcopy(context)
    total_refs = sum(len(item["referencias"]) for item in source["itens"])
    for text_size, ref_size, cap in ((1600, 600, 8), (900, 360, 4), (500, 240, 2), (240, 160, 1), (0, 100, 1), (0, 0, 0)):
        compact = copy.deepcopy(source)
        compact["noticia"]["texto"] = excerpt(source["noticia"]["texto"], text_size) if text_size else ""
        compact["noticia"]["titulo"] = excerpt(source["noticia"]["titulo"], 220)
        included = 0
        for item in compact["itens"]:
            item["texto_selecionado"] = excerpt(item["texto_selecionado"], 220)
            item["sinais"] = sorted({s.get("stance", "") for s in item.pop("registros_automaticos")})
            # Diversidade de resultados/origens antes de repetir a mesma categoria.
            refs, rest, seen = [], [], set()
            for ref in item["referencias"]:
                key = (ref["sinal_automatico"], ref["confianca_na_origem"], ref["somente_contexto"])
                (rest if key in seen else refs).append(ref)
                seen.add(key)
            item["referencias"] = (refs + rest)[:cap]
            for ref in item["referencias"]:
                ref["texto"] = excerpt(ref["texto"], ref_size)
                ref["motivo_do_sinal"] = excerpt(ref.get("motivo_do_sinal"), 140)
                ref["fonte"] = excerpt(ref["fonte"], 60)
                for field in ("id", "url", "titulo", "similaridade_recuperacao", "similaridade_no_sinal"):
                    ref.pop(field, None)
            included += len(item["referencias"])
        compact["recorte"] = {"itens_totais": len(source["itens"]),
                              "referencias_totais": total_refs,
                              "referencias_enviadas": included,
                              "referencias_omitidas": total_refs - included,
                              "textos_podem_ser_abreviados": True}
        if len(json.dumps(compact, ensure_ascii=False, separators=(",", ":"))) <= max_chars:
            return compact
    # Casos com centenas de itens: amostra distribuída, nunca JSON fatiado.
    while compact["itens"]:
        compact["itens"] = compact["itens"][::2] if len(compact["itens"]) > 1 else []
        compact["recorte"]["itens_omitidos"] = len(source["itens"]) - len(compact["itens"])
        if len(json.dumps(compact, ensure_ascii=False, separators=(",", ":"))) <= max_chars:
            return compact
    raise ValueError("Contexto decisório excede o limite configurado; dados não foram cortados silenciosamente.")


def _item_reading(item: dict) -> str:
    """Traduz sinais internos sem promovê-los a fatos confirmados."""
    signals = {
        norm(ref.get("sinal_automatico"))
        for ref in item.get("referencias", [])
        if ref.get("sinal_automatico")
    }
    signals.update(
        norm(record.get("stance"))
        for record in item.get("registros_automaticos", [])
        if record.get("stance")
    )
    has_support = "support" in signals
    has_difference = "contradict" in signals
    if has_support and has_difference:
        return "As comparações automáticas apontaram concordâncias e também diferenças; este ponto ficou parcialmente esclarecido."
    if has_support:
        return "As comparações automáticas encontraram material que aponta na mesma direção, sem transformar essa relação em prova do acontecimento."
    if has_difference:
        return "As comparações automáticas apontaram diferenças em relação aos materiais encontrados; elas podem envolver contexto, data ou formulação."
    if item.get("referencias"):
        return "Foram encontrados materiais relacionados ao tema, mas eles não deram apoio direto suficiente a este ponto."
    return "Não foi encontrada referência suficiente para avaliar este ponto."


def _ordered_items(items: list[dict]) -> list[dict]:
    """Espalha casos favoráveis, divergentes e inconclusivos pelo recorte."""
    groups = {"mixed": [], "support": [], "difference": [], "open": []}
    for item in items:
        signals = {
            norm(ref.get("sinal_automatico"))
            for ref in item.get("referencias", [])
            if ref.get("sinal_automatico")
        }
        signals.update(
            norm(record.get("stance"))
            for record in item.get("registros_automaticos", [])
            if record.get("stance")
        )
        key = ("mixed" if {"support", "contradict"} <= signals else
               "support" if "support" in signals else
               "difference" if "contradict" in signals else "open")
        groups[key].append(item)
    ordered = []
    while any(groups.values()):
        for key in ("support", "difference", "mixed", "open"):
            if groups[key]:
                ordered.append(groups[key].pop(0))
    return ordered


def build_model_context(context: dict, max_chars: int) -> tuple[dict, dict]:
    """Cria uma ficha editorial; campos de diagnóstico não chegam ao modelo.

    O modelo recebe conteúdo suficiente para escrever sobre o caso concreto, mas
    não recebe contagens de recorte, nomes de campos internos ou estados de
    confiança que poderiam ser confundidos com julgamentos sobre uma fonte.
    """
    decision = context["decisao"]
    reason_text = {
        "criterios_para_confiavel_atendidos": "Os materiais aceitos e os sinais de apoio atenderam aos critérios da classificação.",
        "criterios_atendidos_parcialmente": "Houve elementos favoráveis, mas nem todos os critérios da classificação mais alta foram atendidos.",
        "sinais_de_contradicao": "As diferenças apontadas nas comparações atingiram o limite usado para esta classificação.",
        "nenhum_item_aceito_no_calculo": "Nenhum material passou por todos os critérios necessários para sustentar a avaliação.",
        "poucos_itens_aceitos_no_calculo": "Poucos materiais passaram por todos os critérios necessários para sustentar o conteúdo principal.",
        "resultado_inconclusivo": "O conjunto disponível não ofereceu base suficiente para uma conclusão mais segura.",
        "motivo_nao_documentado": "Os dados preservados não registram um único motivo determinante para a classificação.",
    }.get(decision.get("motivo"), "O conjunto disponível não ofereceu base suficiente para uma conclusão mais segura.")

    article = context.get("noticia") or {}
    reputation = context.get("sinais", {}).get("origem", {}).get("nota")
    textual = context.get("sinais", {}).get("texto", {}).get("nota")
    if reputation is not None and reputation >= 80:
        auxiliary = "A boa reputação do veículo ajudou na nota, mas não confirma sozinha as informações da notícia."
    elif reputation is not None and reputation < 50:
        auxiliary = "A avaliação baixa da reputação do veículo limitou esse componente da nota, sem provar que a notícia seja falsa."
    elif reputation is not None:
        auxiliary = "A reputação do veículo participou da nota, mas não confirma sozinha as informações da notícia."
    elif textual is not None and textual >= 80:
        auxiliary = "Os padrões de escrita ajudaram na nota, mas não confirmam os acontecimentos relatados."
    elif textual is not None:
        auxiliary = "Os padrões de escrita participaram da nota apenas como sinal auxiliar."
    else:
        auxiliary = "Não havia um fator auxiliar disponível que pudesse aumentar a segurança da conclusão."
    ordered = _ordered_items(context.get("itens") or [])
    diagnostics = {
        "itens_disponiveis": len(ordered),
        "referencias_disponiveis": sum(len(item.get("referencias", [])) for item in ordered),
        "observacao": "O recorte é apenas técnico e não significa que materiais foram rejeitados pela análise.",
    }

    # Tenta primeiro uma ficha rica; reduz trechos, referências e itens de forma
    # distribuída se o contexto configurado for menor.
    for article_size, claim_size, ref_size, refs_per_item, item_cap in (
        (1200, 220, 220, 2, 12),
        (800, 190, 160, 1, 10),
        (450, 170, 100, 1, 8),
        (240, 150, 0, 0, 8),
        (0, 130, 0, 0, 6),
    ):
        selected = ordered[:item_cap]
        points = []
        for item in selected:
            point = {
                "parte_da_noticia": excerpt(item.get("texto_selecionado", ""), claim_size),
                "leitura_cautelosa": _item_reading(item),
            }
            if refs_per_item:
                snippets = [excerpt(ref.get("texto", ""), ref_size)
                            for ref in item.get("referencias", []) if plain(ref.get("texto"))]
                if snippets:
                    point["trechos_relacionados"] = snippets[:refs_per_item]
            points.append(point)
        brief = {
            "resultado_que_deve_ser_explicado": decision.get("rotulo"),
            "motivo_principal": reason_text,
            "fator_auxiliar_da_nota": auxiliary,
            "assunto": excerpt(article.get("titulo", ""), 220),
            "contexto_da_noticia": excerpt(article.get("texto", ""), article_size) if article_size else "",
            "pontos_avaliados": points,
            "limites_da_conclusao": [
                "Uma comparação automática pode apontar relação ou diferença sem comprovar o acontecimento.",
                "Não encontrar apoio suficiente deixa o ponto em aberto; não prova que a notícia seja falsa.",
                "A reputação do veículo e os padrões de escrita são sinais auxiliares, não provas dos fatos.",
            ],
        }
        encoded = json.dumps(brief, ensure_ascii=False, separators=(",", ":"))
        if len(encoded) <= max_chars:
            diagnostics.update({
                "itens_enviados": len(points),
                "referencias_textuais_enviadas": sum(len(p.get("trechos_relacionados", [])) for p in points),
                "caracteres_enviados": len(encoded),
            })
            return brief, diagnostics
    raise ValueError("A ficha editorial excede o limite configurado.")


def restore_result(payload: dict, content: str = "") -> SimpleNamespace:
    """Restaura o snapshot salvo, sem buscar evidências nem usar o RAG atual."""
    analysis = payload.get("analysis") or {}
    retrievals = []
    for item in (payload.get("evidence") or {}).get("claims", []):
        retrievals.append(SimpleNamespace(
            claim=SimpleNamespace(claim_id=item.get("claim_id", ""), text=item.get("text", "")),
            evidences=[SimpleNamespace(**ev) for ev in item.get("evidences", [])],
        ))
    return SimpleNamespace(
        title=analysis.get("title", ""), url=analysis.get("url", ""), content=content,
        label_final=analysis.get("label"), score_final=analysis.get("score"),
        score_breakdown=payload.get("analysis", {}).get("score_breakdown") or {},
        retrieval_results=retrievals,
        stance_results=(payload.get("transparency") or {}).get("stance_results") or [],
    )
