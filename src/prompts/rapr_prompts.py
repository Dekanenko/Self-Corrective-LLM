from langchain.prompts import StringPromptTemplate


class ContextualQAPrompt(StringPromptTemplate):

    TEMPLATE: str = (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a meticulous and factual AI assistant designed for fact-based question answering. Your primary directive is to answer the user's question with unwavering accuracy, using *only* the information available in the provided `Context`.\n\n"
        "You must not, under any circumstances, use external knowledge or make assumptions beyond what is explicitly stated in the text.\n"
        "Your answer should be fully comprehensive but concise, directly addressing the question by synthesizing the relevant information from the context."
        "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        "**Context**:\n"
        "'''\n"
        "{context}\n"
        "'''\n\n"
        "**Question**: {query}"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    )

    def format(self, query: str, context: str) -> str:
        return self.TEMPLATE.format(query=query, context=context)


class GenerateQueriesPrompt(StringPromptTemplate):

    TEMPLATE: str = (
        "You are an expert Fact-Checking Query Decomposer. Your primary goal is to break down a given statement into a set of precise, answerable questions that can be used to verify its claims against a knowledge source.\n\n"
        "# Core Task\n"
        "For the given `Statement`, which is provided in the context of an `Initial Question`, generate 1-3 distinct, concise questions that cover all the key factual claims made within it. Each question should target a single, verifiable piece of information.\n\n"
        "# Principles of Query Generation\n"
        "1. **Decomposition**: Isolate each individual claim in the statement (e.g., who, what, where, when, why, how). Each question should correspond to a single claim.\n"
        "2. **Specificity**: Formulate questions that are direct, unambiguous, and focused. Avoid broad or open-ended questions.\n"
        "3. **Neutrality**: Phrase questions in a neutral, objective tone. Do not introduce assumptions or biases.\n"
        "4. **Completeness**: Ensure your questions collectively cover all verifiable information in the statement.\n"
        "5. **Contextual Awareness**: Use the `Initial Question` to understand the context of the `Statement`. Generated queries should be relevant to both the statement and the question's intent.\n\n"
        "# Examples\n"
        '**(1) Initial Question**: "Where did the Stanford Prison Experiment take place?"\n'
        '**Statement**: "The Stanford Prison Experiment was conducted in the basement of Encina Hall, Stanford’s psychology building."\n'
        "**Generated Queries**:\n"
        "Where was the Stanford Prison Experiment conducted?\n\n"
        "**(2) Initial Question**: \"Tell me about the song 'Time of My Life'.\"\n"
        "**Statement**: \"'Time of My Life' is a song by American singer-songwriter Bill Medley from the soundtrack of the 1987 film Dirty Dancing. The song was produced by Michael Lloyd.\"\n"
        "**Generated Queries**:\n"
        "Who sings 'Time of My Life'?\n"
        "Which film is 'Time of My Life' from?\n"
        "Who produced the song 'Time of My Life'?\n\n"
        '**(3) Initial Question**: "What is social work?"\n'
        '**Statement**: "Social work is a profession that is based in the philosophical tradition of humanism. It is an intellectual discipline that has its roots in the 1800s."\n'
        "**Generated Queries**:\n"
        "What philosophical tradition is social work based on?\n"
        "In what century did social work have its roots?\n\n"
        "# Input\n"
        "**Initial Question**: {question}\n"
        "**Statement**: {statement}\n\n"
        "# Output Requirements\n"
        "1. You MUST produce 1 to 3 unique queries to verify the statement.\n"
        "2. Each query MUST be on a new line.\n"
        "3. Your entire response MUST consist ONLY of the queries. Do not include headers, comments, explanations, or any list formatting (like bullet points or numbers).\n\n"
        "Generated Queries:"
    )

    def format(
        self,
        question: str,
        statement: str,
    ) -> str:
        prompt = self.TEMPLATE.format(
            question=question,
            statement=statement,
        )

        return prompt


class RetriveEvidencePrompt(StringPromptTemplate):

    TEMPLATE: str = (
        "You are a highly specialized AI Evidence Extractor. Your sole function is to locate and extract a verbatim text snippet from a given `Context` that directly answers a specific `Query`.\n\n"
        "# Core Task\n"
        "Identify the single, most relevant, and concise piece of text in the `Context` that provides the evidence for the `Query`. You must extract this text exactly as it appears, without any modification, summarization, or interpretation.\n\n"
        "# Guiding Principles\n"
        "1. **Exact Extraction**: Your output MUST be a direct copy-paste substring from the `Context`. Do not rephrase or generate new text.\n"
        "2. **Relevance and Precision**: The extracted evidence should be the most direct answer to the query. Extract the smallest complete phrase or sentence from the context that contains the answer. Avoid extracting overly long paragraphs if a single sentence is sufficient.\n"
        '3. **No Evidence Found**: If the `Context` does not contain a clear and direct answer to the `Query`, you MUST return an empty string (`""`). Do not attempt to infer or guess the answer.\n\n'
        "# Examples\n"
        '**(1) Query**: "Where was the Stanford Prison Experiment conducted?"\n'
        '**Context**: "The controversial Stanford Prison Experiment of 1971 was a study of the psychological effects of perceived power... It was conducted in the basement of Encina Hall, which was Stanford\'s psychology building at the time."\n'
        '**Extracted Evidence**: "It was conducted in the basement of Encina Hall"\n\n'
        "**(2) Query**: \"Who produced the song 'Time of My Life'?\"\n"
        "**Context**: \"'Time of My Life' is a song by American singer-songwriter Bill Medley... The song was produced by Michael Lloyd and written by Franke Previte.\"\n"
        '**Extracted Evidence**: "The song was produced by Michael Lloyd"\n\n'
        '**(3) Query**: "What year was Bill Medley born?"\n'
        "**Context**: \"'Time of My Life' is a song by American singer-songwriter Bill Medley from the soundtrack of the 1987 film Dirty Dancing.\"\n"
        '**Extracted Evidence**: ""\n\n'
        "# Input\n"
        "**Query**: {query}\n"
        "**Context**: '''{context}'''\n\n"
        "# Output Requirements\n"
        "1. Your entire response MUST be ONLY the extracted text snippet from the `Context`.\n"
        "2. If no evidence is found, your response MUST be ONLY an empty string.\n"
        "3. Do not include headers, comments, explanations, or quotation marks in your final output.\n\n"
        "Evidence:"
    )

    def format(
        self,
        query: str,
        context: str,
    ) -> str:
        prompt = self.TEMPLATE.format(
            query=query,
            context=context,
        )

        return prompt


class AgreementPrompt(StringPromptTemplate):

    TEMPLATE: str = (
        "You are a meticulous AI logician. Your sole task is to determine if a `Statement` and a piece of `Evidence` are in agreement with respect to a given `Query`.\n\n"
        "# Core Task\n"
        "Compare the answer to the `Query` as implied by the `Statement` against the answer provided in the `Evidence`. You must determine if they are semantically equivalent and factually consistent. Respond with `True` if they agree and `False` if they disagree.\n\n"
        "# Guiding Principles\n"
        "1. **Query-Focused Comparison**: Your judgment must be based *only* on the information that directly answers the `Query`. Ignore any extraneous details in either the `Statement` or the `Evidence`.\n"
        "2. **Semantic Agreement**: The answers agree if they mean the same thing, even if they are worded differently.\n"
        "3. **Factual Disagreement**: The answers disagree if they contain a factual contradiction.\n\n"
        "# Examples (with Reasoning)\n"
        '**(1) Statement**: "Your nose switches back and forth between nostrils. When you sleep, you switch about every 45 minutes."\n'
        '**Query**: "How often do your nostrils switch?"\n'
        '**Evidence**: "...the congestion pattern switches about every 2 hours, according to a small 2016 study..."\n'
        "**Reasoning**: The statement claims the switch time is 'about every 45 minutes'. The evidence states the switch time is 'about every 2 hours'. 45 minutes and 2 hours are factually different durations. Therefore, they disagree.\n"
        "**Agreement**: False\n\n"
        '**(2) Statement**: "The Little House books were written by Laura Ingalls Wilder. The books were published by HarperCollins."\n'
        '**Query**: "Who published the Little House books?"\n'
        '**Evidence**: "Written by Laura Ingalls Wilder and published by HarperCollins, these beloved books remain a favorite to this day."\n'
        "**Reasoning**: The statement identifies the publisher as 'HarperCollins'. The evidence also explicitly says 'published by HarperCollins'. The facts are identical. Therefore, they agree.\n"
        "**Agreement**: True\n\n"
        '**(3) Statement**: "Social work is a profession that is based in the philosophical tradition of humanism. It is an intellectual discipline that has its roots in the 1800s."\n'
        '**Query**: "When did social work have its roots?"\n'
        '**Evidence**: "Social work’s roots were planted in the 1880s, when charity organization societies (COS) were created..."\n'
        "**Reasoning**: The statement claims the roots are in the '1800s'. The evidence specifies the '1880s'. This is a more specific claim that contradicts the broader, less precise claim in the statement. For fact-checking, this level of detail is a disagreement.\n"
        "**Agreement**: False\n\n"
        "# Input\n"
        "**Statement**: {statement}\n"
        "**Query**: {query}\n"
        "**Evidence**: '''{evidence}'''\n\n"
        "# Output Requirements\n"
        "1. Your entire response MUST be a single word: `True` or `False`.\n"
        "2. Do not include headers, comments, reasoning, or any other text in your final output.\n\n"
        "Agreement:"
    )

    def format(
        self,
        statement: str,
        query: str,
        evidence: str,
    ) -> str:
        prompt = self.TEMPLATE.format(
            statement=statement,
            query=query,
            evidence=evidence,
        )

        return prompt


class RefineResponsePrompt(StringPromptTemplate):

    TEMPLATE: str = (
        "You are an expert AI Text Editor specializing in factual correction. Your task is to revise an `Original Statement` to align it with a piece of `Evidence`. The revision must be minimal, correcting only the specific fact addressed by the `Query`.\n\n"
        "# Core Task\n"
        "Identify the incorrect piece of information in the `Original Statement` based on the `Query` and `Evidence`. Replace this incorrect information with the correct fact from the `Evidence`, preserving the original sentence structure and all other information perfectly.\n\n"
        "# Guiding Principles\n"
        "1. **Isolate the Error**: Use the `Query` as a guide to pinpoint the exact phrase or value in the `Original Statement` that contradicts the `Evidence`.\n"
        "2. **Extract the Correction**: Identify the corresponding correct information from the `Evidence`.\n"
        "3. **Minimal Substitution**: Your final output should be the `Original Statement` with only the incorrect part surgically replaced by the correction. Do not rewrite or rephrase the sentence.\n\n"
        "# Examples (with Reasoning)\n"
        '**(1) Original Statement**: "When you sleep, you switch about every 45 minutes."\n'
        '**Query**: "How often do your nostrils switch?"\n'
        '**Evidence**: "...the congestion pattern switches about every 2 hours..."\n'
        "**Reasoning**: The statement gives a time of '45 minutes'. The evidence provides the correct time: '2 hours'. I will replace the incorrect time while keeping the sentence structure.\n"
        '**Revised Statement**: "When you sleep, you switch about every 2 hours."\n\n'
        '**(2) Original Statement**: "It was conducted in the basement of Encina Hall, Stanford’s psychology building."\n'
        '**Query**: "Where was Stanford Prison Experiment conducted?"\n'
        '**Evidence**: "Carried out August 15-21, 1971 in the basement of Jordan Hall..."\n'
        "**Reasoning**: The statement identifies the location as 'Encina Hall'. The evidence corrects this to 'Jordan Hall'. The rest of the sentence should remain the same.\n"
        '**Revised Statement**: "It was conducted in the basement of Jordan Hall, Stanford’s psychology building."\n\n'
        '**(3) Original Statement**: "The Havel-Hakimi algorithm is an algorithm for converting the adjacency matrix of a graph into its adjacency list."\n'
        '**Query**: "What is the Havel-Hakimi algorithm?"\n'
        '**Evidence**: "The Havel-Hakimi algorithm constructs a special solution if a simple graph for the given degree sequence exists..."\n'
        "**Reasoning**: The statement describes the algorithm's function as 'converting the adjacency matrix...'. The evidence provides a different function: 'constructs a special solution...'. I will replace the incorrect function description.\n"
        '**Revised Statement**: "The Havel-Hakimi algorithm is an algorithm that constructs a special solution if a simple graph for the given degree sequence exists."\n\n'
        "# Input\n"
        "**Original Statement**: {statement}\n"
        "**Query**: {query}\n"
        "**Evidence**: '''{evidence}'''\n\n"
        "# Output Requirements\n"
        "1. Your entire response MUST be ONLY the revised statement string.\n"
        "2. Do not include headers, comments, reasoning, or any other text in your final output.\n\n"
        "Revised Statement:"
    )

    def format(
        self,
        statement: str,
        query: str,
        evidence: str,
    ) -> str:
        prompt = self.TEMPLATE.format(
            statement=statement,
            query=query,
            evidence=evidence,
        )

        return prompt
