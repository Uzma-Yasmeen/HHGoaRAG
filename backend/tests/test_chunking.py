from app.chunking import multi_strategy_chunks
def test_multiple_strategies_and_deduplication():
    text=("One useful sentence explains a fact clearly. Another sentence adds enough supporting context. A third sentence gives a relevant example. ")*8
    chunks=multi_strategy_chunks(text,"en",1,True);strategies={c.strategy for c in chunks}
    assert "native_passage" in strategies and "sentence_window_2_overlap_1" in strategies and "word_120_overlap_30" in strategies and "answer_centred" in strategies
    assert len({c.text.casefold() for c in chunks})==len(chunks)
