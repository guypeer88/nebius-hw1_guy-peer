def build_system_prompt() -> str:
    return (
        "You write concise e-commerce product descriptions.\n\n"
        "Write a product description that:\n"
        "- is between 50 and 90 words\n" # length
        "- uses correct grammar, spelling, punctuation, and capitalization\n" # grammar
        "- reads naturally, with clear sentence flow and no awkward phrasing or unnecessary repetition\n" # fluency
        "- uses a friendly, credible, and moderately persuasive sales tone\n" # tone
        "- stays grounded in the provided product information\n\n" # grounding
        "You may use mild marketing language, but do not invent features, "
        "performance claims, quality claims, or emotional promises that are "
        "not supported by the provided information.\n\n"
        "If the product information is limited, keep the description simple and "
        "accurate rather than filling gaps with assumptions."
    )

def build_user_prompt(product: dict) -> str:
    return f"""Write a concise e-commerce product description using only the information below.

            Product name: {product.get('name', '')}
            Product details: {product.get('attributes', '')}
            Material: {product.get('material', '')}
            Warranty: {product.get('warranty', '')}

            Do not add facts, features, performance claims, or quality claims that are not supported by the provided information.
            """

def build_stricter_grounding_system_prompt() -> str:
    return (
        "You write concise e-commerce product descriptions.\n\n"
        "Write a product description that:\n"
        "- is between 50 and 90 words\n"
        "- uses correct grammar, spelling, punctuation, and capitalization\n"
        "- reads naturally, with clear sentence flow and no awkward phrasing or unnecessary repetition\n"
        "- uses a friendly, credible, and moderately persuasive sales tone\n"
        "- stays strictly grounded in the provided product information\n\n"
        "Use only facts explicitly stated in the input.\n"
        "Do not invent or exaggerate features, benefits, performance claims, durability claims, quality claims, or emotional promises.\n"
        "Do not use unsupported adjectives such as premium, durable, powerful, high-quality, or built to last unless the input clearly supports them.\n"
        "If the information is limited, keep the description simple and accurate rather than filling gaps with assumptions.\n\n"
        "Return only the final product description as plain text."
    )