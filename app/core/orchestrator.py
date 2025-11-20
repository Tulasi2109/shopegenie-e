# app/core/orchestrator.py

def run_pipeline(user_query: str):
    """
    Runs the core ShopGenie-E pipeline:
    1) Parse intent
    2) Filter candidate products
    3) Rank products with explanations
    """

    # Lazy imports to avoid circular imports
    from agents.intent_agent import extract_intent
    from agents.retrieval_agent import filter_products
    from agents.reasoner_agent import rank_products

    # 1. Intent extraction
    intent = extract_intent(user_query)

    # 2. Product retrieval and filtering (now uses both intent + raw query)
    products = filter_products(intent, user_query)

    # 3. Reasoning and ranking
    ranking = rank_products(intent, products)

    return intent, products, ranking


if __name__ == "__main__":
    # Example query to test the pipeline end-to-end
    query = "Best laptop for data science under 900 dollars with at least 16 GB RAM and good battery life."

    from agents.intent_agent import extract_intent
    from agents.retrieval_agent import filter_products
    from agents.reasoner_agent import rank_products

    intent = extract_intent(query)
    products = filter_products(intent, query)
    ranking = rank_products(intent, products)

    print("\n=== USER QUERY ===")
    print(query)

    print("\n=== PARSED INTENT ===")
    print(intent)

    print("\n=== CANDIDATE PRODUCTS (after filters) ===")
    print(products)

    print("\n=== RANKING RESULT ===")
    print(ranking)
