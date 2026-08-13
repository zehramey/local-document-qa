"""Streamlit front-end for local-document-qa.

Talks only to the FastAPI backend over HTTP — no direct access to Qdrant,
the embedding model, or the LLM. Every backend error is shown as the
clean {error_code, message} the API returns, never a raw exception or
traceback.
"""

import os
import time
from typing import Any

import httpx
import streamlit as st

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")


def _api_error_message(response: httpx.Response) -> str:
    try:
        body = response.json()
        return str(body.get("message") or body.get("detail") or response.text)
    except ValueError:
        return f"Sunucu hatası (HTTP {response.status_code})"


def _get(path: str, **kwargs: object) -> httpx.Response:
    return httpx.get(f"{API_BASE_URL}{path}", timeout=30.0, **kwargs)  # type: ignore[arg-type]


def _post(path: str, **kwargs: object) -> httpx.Response:
    # Must stay above LLM_TIMEOUT_SECONDS (.env) so a slow LLM gets to return
    # its own {error_code, message} instead of the UI giving up first.
    return httpx.post(f"{API_BASE_URL}{path}", timeout=220.0, **kwargs)  # type: ignore[arg-type]


def _delete(path: str) -> httpx.Response:
    return httpx.delete(f"{API_BASE_URL}{path}", timeout=30.0)


st.set_page_config(page_title="Local Document QA", layout="wide")
st.title("Local Document QA")

try:
    health = _get("/health")
    if health.status_code != 200:
        st.error(f"API sağlıklı değil: {_api_error_message(health)}")
except httpx.HTTPError as exc:
    st.error(f"API'ye ({API_BASE_URL}) bağlanılamadı: {exc}")
    st.stop()

# -- 1. Upload -----------------------------------------------------------

st.header("1. Doküman Yükle")
uploaded_file = st.file_uploader("PDF veya TXT dosyası seçin", type=["pdf", "txt"])

if uploaded_file is not None and st.button("Yükle ve İndeksle"):
    with st.spinner("İndeksleniyor..."):
        try:
            response = _post(
                "/documents",
                files={"file": (uploaded_file.name, uploaded_file.getvalue())},
            )
        except httpx.HTTPError as exc:
            st.error(f"Yükleme başarısız: bağlantı hatası ({exc})")
        else:
            if response.status_code == 201:
                body = response.json()
                st.success(
                    f"'{body['filename']}' indeksledi: {body['chunk_count']} chunk "
                    f"(document_id: {body['document_id']})"
                )
            else:
                st.error(f"Yükleme başarısız: {_api_error_message(response)}")

# -- 2/3/4. Document list + active document selection ---------------------

st.header("2. Yüklenen Dokümanlar")
try:
    documents_response = _get("/documents")
except httpx.HTTPError as exc:
    st.error(f"Doküman listesi alınamadı: bağlantı hatası ({exc})")
    documents = []
else:
    if documents_response.status_code == 200:
        documents = documents_response.json()["documents"]
    else:
        st.error(f"Doküman listesi alınamadı: {_api_error_message(documents_response)}")
        documents = []

if not documents:
    st.info("Henüz bir doküman yüklenmedi.")
    st.stop()

document_labels = {
    f"{doc['filename']} ({doc['chunk_count']} chunk, {doc['page_count']} sayfa)": doc["document_id"]
    for doc in documents
}
selected_label = st.selectbox("Aktif doküman", list(document_labels))
active_document_id = document_labels[selected_label]

if st.button("Seçili dokümanı sil"):
    delete_response = _delete(f"/documents/{active_document_id}")
    if delete_response.status_code == 204:
        st.success("Doküman silindi.")
        st.rerun()
    else:
        st.error(f"Silme başarısız: {_api_error_message(delete_response)}")

# -- 5. LLM selection ------------------------------------------------------

st.header("3. Model Seçimi")
try:
    models_response = _get("/models")
except httpx.HTTPError as exc:
    st.warning(f"Model listesi alınamadı (varsayılan model kullanılacak): {exc}")
    model_ids: list[str] = []
else:
    if models_response.status_code == 200:
        all_models = models_response.json()["models"]
        model_ids = [m["model_id"] for m in all_models if m["model_type"] != "embeddings"]
    else:
        st.warning(f"Model listesi alınamadı: {_api_error_message(models_response)}")
        model_ids = []

selected_model_ids = (
    st.multiselect(
        "LLM (karşılaştırmak için birden fazla model seçebilirsiniz)",
        model_ids,
        default=model_ids[:1],
    )
    if model_ids
    else []
)


def _render_answer_result(result: dict[str, Any]) -> None:
    if result["answerable"]:
        st.success(result["answer"])
    else:
        st.warning(result["answer"])
    st.caption(f"Answerable: {result['answerable']}")

    if result["rejected_citation_ids"]:
        st.warning(
            "Model bazı kaynakları uydurdu ve bu kaynaklar reddedildi: "
            f"{', '.join(result['rejected_citation_ids'])}"
        )

    if result["citations"]:
        st.subheader("Kaynaklar")
        for citation in result["citations"]:
            st.markdown(
                f"- **{citation['filename']}**, sayfa "
                f"{citation['page_start']}-{citation['page_end']} "
                f"(chunk_id: `{citation['chunk_id'][:12]}...`)"
            )

    with st.expander("Kullanılan context'i göster/gizle"):
        for chunk in result["retrieved_chunks"]:
            reranker_score = chunk["reranker_score"]
            reranker_text = f"{reranker_score:.3f}" if reranker_score is not None else "—"
            st.markdown(
                f"**#{chunk['final_rank']}** — sayfa {chunk['page_start']}-"
                f"{chunk['page_end']} — retrieval_score: {chunk['retrieval_score']:.3f} — "
                f"reranker_score: {reranker_text}"
            )
            st.text(chunk["text"])
            st.divider()

    metrics = result["metrics"]
    if metrics is not None:
        st.caption(
            f"Model: {metrics['model_id']} (quantization: {metrics['quantization']}) — "
            f"cevap süresi: {metrics['total_duration_seconds']:.2f}s — "
            f"prompt_tokens={metrics['prompt_tokens']} "
            f"output_tokens={metrics['output_tokens']}"
        )


# -- 6/7/8/9/10/11/12/13/14/15/16. Question + answer ----------------------

st.header("4. Soru Sor")
question = st.text_input("Sorunuz")

if st.button("Sor") and question:
    if not selected_model_ids:
        st.error("En az bir model seçin.")
    else:
        results: dict[str, dict[str, Any]] = {}
        errors: dict[str, str] = {}
        client_elapsed: dict[str, float] = {}

        for model_id in selected_model_ids:
            with st.spinner(f"'{model_id}' cevap üretiyor..."):
                start = time.monotonic()
                try:
                    response = _post(
                        "/questions",
                        json={
                            "document_id": active_document_id,
                            "question": question,
                            "model_id": model_id,
                        },
                    )
                except httpx.HTTPError as exc:
                    errors[model_id] = f"bağlantı hatası ({exc})"
                    continue
                client_elapsed[model_id] = time.monotonic() - start

            if response.status_code != 200:
                errors[model_id] = _api_error_message(response)
            else:
                results[model_id] = response.json()

        # Comparison table only makes sense with 2+ models; with one, it'd
        # just repeat the single answer already rendered below in its tab.
        if len(selected_model_ids) > 1:
            st.subheader("Karşılaştırma")
            st.dataframe(
                [
                    {
                        "Model": model_id,
                        "Durum": "Hata" if model_id in errors else "OK",
                        "Cevaplanabilir": (
                            results[model_id]["answerable"] if model_id in results else "—"
                        ),
                        "Kaynak sayısı": (
                            len(results[model_id]["citations"]) if model_id in results else "—"
                        ),
                        "Reddedilen kaynak": (
                            len(results[model_id]["rejected_citation_ids"])
                            if model_id in results
                            else "—"
                        ),
                        "Sunucu süresi (s)": (
                            round(results[model_id]["metrics"]["total_duration_seconds"], 2)
                            if model_id in results and results[model_id]["metrics"]
                            else "—"
                        ),
                        "token/s": (
                            round(results[model_id]["metrics"]["tokens_per_second"], 2)
                            if model_id in results
                            and results[model_id]["metrics"]
                            and results[model_id]["metrics"]["tokens_per_second"]
                            else "—"
                        ),
                        "İstemci süresi (s)": round(client_elapsed.get(model_id, 0.0), 2),
                    }
                    for model_id in selected_model_ids
                ],
                use_container_width=True,
                hide_index=True,
            )

        tabs = st.tabs(selected_model_ids)
        for tab, model_id in zip(tabs, selected_model_ids, strict=True):
            with tab:
                if model_id in errors:
                    st.error(f"Hata: {errors[model_id]}")
                else:
                    _render_answer_result(results[model_id])
                st.caption(
                    f"İstemci tarafında ölçülen süre: {client_elapsed.get(model_id, 0.0):.2f}s"
                )
