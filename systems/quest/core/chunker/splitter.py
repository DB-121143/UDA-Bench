"""Experimental **text splitter** based on semantic similarity."""
from __future__ import annotations

import copy
import re
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple, cast

import numpy as np
from langchain_community.utils.math import (
    cosine_similarity,
)
from langchain_core.documents import BaseDocumentTransformer, Document
from langchain_core.embeddings import Embeddings



from typing import Any, List

from langchain_text_splitters.base import TextSplitter

from quest.conf.settings import Enc_token_cnt


import re
import tiktoken
from typing import Any, List, Literal, Optional, Union

from langchain_text_splitters.base import Language, TextSplitter


def _split_text_with_regex(
    text: str, separator: str, keep_separator: Union[bool, Literal["start", "end"]]
) -> List[str]:
    # Now that we have the separator, split the text
    if separator:
        if keep_separator:
            # The parentheses in the pattern keep the delimiters in the result.
            _splits = re.split(f"({separator})", text)
            splits = (
                ([_splits[i] + _splits[i + 1] for i in range(0, len(_splits) - 1, 2)])
                if keep_separator == "end"
                else ([_splits[i] + _splits[i + 1] for i in range(1, len(_splits), 2)])
            )
            if len(_splits) % 2 == 0:
                splits += _splits[-1:]
            splits = (
                (splits + [_splits[-1]])
                if keep_separator == "end"
                else ([_splits[0]] + splits)
            )
        else:
            splits = re.split(separator, text)
    else:
        splits = list(text)
    return [s for s in splits if s != ""]


class RecursiveTokenTextSplitter(TextSplitter):
    """Splitting text by recursively look at characters, but using token count for chunk size.

    Recursively tries to split by different characters to find one
    that works, while using token-based counting for determining chunk sizes.
    """

    def __init__(
        self,
        separators: Optional[List[str]] = ["\n\n", ".", "\n", " ", ""],
        keep_separator: Union[bool, Literal["start", "end"]] = True,
        is_separator_regex: bool = False,
        **kwargs: Any,
    ) -> None:
        """Create a new TokenTextSplitter."""
        super().__init__(keep_separator=keep_separator, **kwargs)
        self._separators = separators or ["\n\n", "\n", " ", ""]
        self._is_separator_regex = is_separator_regex
        # Initialize the tokenizer
        self._tokenizer = tiktoken.get_encoding("cl100k_base")

    def _token_length_function(self, text: str) -> int:
        """Return the number of tokens in the text."""
        return len(self._tokenizer.encode(text))

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Split incoming text and return chunks."""
        final_chunks = []
        # Get appropriate separator to use
        separator = separators[-1]
        new_separators = []
        for i, _s in enumerate(separators):
            _separator = _s if self._is_separator_regex else re.escape(_s)
            if _s == "":
                separator = _s
                break
            if re.search(_separator, text):
                separator = _s
                new_separators = separators[i + 1 :]
                break

        _separator = separator if self._is_separator_regex else re.escape(separator)
        splits = _split_text_with_regex(text, _separator, self._keep_separator)

        # Now go merging things, recursively splitting longer texts.
        _good_splits = []
        _separator = "" if self._keep_separator else separator
        for s in splits:
            if self._token_length_function(s) < self._chunk_size:
                _good_splits.append(s)
            else:
                if _good_splits:
                    merged_text = self._merge_splits(_good_splits, _separator)
                    final_chunks.extend(merged_text)
                    _good_splits = []
                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
        if _good_splits:
            merged_text = self._merge_splits(_good_splits, _separator)
            final_chunks.extend(merged_text)
        return final_chunks

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge splits using token-based length calculation."""
        docs = []
        current_doc = []
        total_tokens = 0
        for d in splits:
            _len = self._token_length_function(d)
            if (
                total_tokens + _len + (len(current_doc) * self._token_length_function(separator) if separator else 0)
                > self._chunk_size
            ):
                if current_doc:
                    doc = separator.join(current_doc)
                    if doc:
                        docs.append(doc)
                    current_doc = [d]
                    total_tokens = _len
                else:
                    current_doc = [d]
                    total_tokens = _len
            else:
                current_doc.append(d)
                total_tokens += _len
        if current_doc:
            doc = separator.join(current_doc)
            if doc:
                docs.append(doc)
        return docs

    def split_text(self, text: str) -> List[str]:
        """Split the input text into smaller chunks based on token count and predefined separators.

        Args:
            text (str): The input text to be split.

        Returns:
            List[str]: A list of text chunks obtained after splitting.
        """
        return self._split_text(text, self._separators)

    @classmethod
    def from_language(
        cls, language: Language, **kwargs: Any
    ) -> "RecursiveTokenTextSplitter":
        """Return an instance of this class based on a specific language.

        This method initializes the text splitter with language-specific separators.

        Args:
            language (Language): The language to configure the text splitter for.
            **kwargs (Any): Additional keyword arguments to customize the splitter.

        Returns:
            RecursiveTokenTextSplitter: An instance of the text splitter configured
            for the specified language.
        """
        separators = cls.get_separators_for_language(language)
        return cls(separators=separators, is_separator_regex=True, **kwargs)

    @staticmethod
    def get_separators_for_language(language: Language) -> List[str]:
        """Retrieve a list of separators specific to the given language.

        Args:
            language (Language): The language for which to get the separators.

        Returns:
            List[str]: A list of separators appropriate for the specified language.
        """
        if language == Language.C or language == Language.CPP:
            return [
                # Split along class definitions
                "\nclass ",
                # Split along function definitions
                "\nvoid ",
                "\nint ",
                "\nfloat ",
                "\ndouble ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                "\ncase ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.GO:
            return [
                # Split along function definitions
                "\nfunc ",
                "\nvar ",
                "\nconst ",
                "\ntype ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nswitch ",
                "\ncase ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.JAVA:
            return [
                # Split along class definitions
                "\nclass ",
                # Split along method definitions
                "\npublic ",
                "\nprotected ",
                "\nprivate ",
                "\nstatic ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                "\ncase ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.KOTLIN:
            return [
                # Split along class definitions
                "\nclass ",
                # Split along method definitions
                "\npublic ",
                "\nprotected ",
                "\nprivate ",
                "\ninternal ",
                "\ncompanion ",
                "\nfun ",
                "\nval ",
                "\nvar ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nwhen ",
                "\ncase ",
                "\nelse ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.JS:
            return [
                # Split along function definitions
                "\nfunction ",
                "\nconst ",
                "\nlet ",
                "\nvar ",
                "\nclass ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                "\ncase ",
                "\ndefault ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.TS:
            return [
                "\nenum ",
                "\ninterface ",
                "\nnamespace ",
                "\ntype ",
                # Split along class definitions
                "\nclass ",
                # Split along function definitions
                "\nfunction ",
                "\nconst ",
                "\nlet ",
                "\nvar ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                "\ncase ",
                "\ndefault ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.PHP:
            return [
                # Split along function definitions
                "\nfunction ",
                # Split along class definitions
                "\nclass ",
                # Split along control flow statements
                "\nif ",
                "\nforeach ",
                "\nwhile ",
                "\ndo ",
                "\nswitch ",
                "\ncase ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.PROTO:
            return [
                # Split along message definitions
                "\nmessage ",
                # Split along service definitions
                "\nservice ",
                # Split along enum definitions
                "\nenum ",
                # Split along option definitions
                "\noption ",
                # Split along import statements
                "\nimport ",
                # Split along syntax declarations
                "\nsyntax ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.PYTHON:
            return [
                # First, try to split along class definitions
                "\nclass ",
                "\ndef ",
                "\n\tdef ",
                # Now split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.RST:
            return [
                # Split along section titles
                "\n=+\n",
                "\n-+\n",
                "\n\\*+\n",
                # Split along directive markers
                "\n\n.. *\n\n",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.RUBY:
            return [
                # Split along method definitions
                "\ndef ",
                "\nclass ",
                # Split along control flow statements
                "\nif ",
                "\nunless ",
                "\nwhile ",
                "\nfor ",
                "\ndo ",
                "\nbegin ",
                "\nrescue ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.ELIXIR:
            return [
                # Split along method function and module definition
                "\ndef ",
                "\ndefp ",
                "\ndefmodule ",
                "\ndefprotocol ",
                "\ndefmacro ",
                "\ndefmacrop ",
                # Split along control flow statements
                "\nif ",
                "\nunless ",
                "\nwhile ",
                "\ncase ",
                "\ncond ",
                "\nwith ",
                "\nfor ",
                "\ndo ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.RUST:
            return [
                # Split along function definitions
                "\nfn ",
                "\nconst ",
                "\nlet ",
                # Split along control flow statements
                "\nif ",
                "\nwhile ",
                "\nfor ",
                "\nloop ",
                "\nmatch ",
                "\nconst ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.SCALA:
            return [
                # Split along class definitions
                "\nclass ",
                "\nobject ",
                # Split along method definitions
                "\ndef ",
                "\nval ",
                "\nvar ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nmatch ",
                "\ncase ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.SWIFT:
            return [
                # Split along function definitions
                "\nfunc ",
                # Split along class definitions
                "\nclass ",
                "\nstruct ",
                "\nenum ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\ndo ",
                "\nswitch ",
                "\ncase ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.MARKDOWN:
            return [
                # First, try to split along Markdown headings (starting with level 2)
                "\n#{1,6} ",
                # Note the alternative syntax for headings (below) is not handled here
                # Heading level 2
                # ---------------
                # End of code block
                "```\n",
                # Horizontal lines
                "\n\\*\\*\\*+\n",
                "\n---+\n",
                "\n___+\n",
                # Note that this splitter doesn't handle horizontal lines defined
                # by *three or more* of ***, ---, or ___, but this is not handled
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.LATEX:
            return [
                # First, try to split along Latex sections
                "\n\\\\chapter{",
                "\n\\\\section{",
                "\n\\\\subsection{",
                "\n\\\\subsubsection{",
                # Now split by environments
                "\n\\\\begin{enumerate}",
                "\n\\\\begin{itemize}",
                "\n\\\\begin{description}",
                "\n\\\\begin{list}",
                "\n\\\\begin{quote}",
                "\n\\\\begin{quotation}",
                "\n\\\\begin{verse}",
                "\n\\\\begin{verbatim}",
                # Now split by math environments
                "\n\\\\begin{align}",
                "$$",
                "$",
                # Now split by the normal type of lines
                " ",
                "",
            ]
        elif language == Language.HTML:
            return [
                # First, try to split along HTML tags
                "<body",
                "<div",
                "<p",
                "<br",
                "<li",
                "<h1",
                "<h2",
                "<h3",
                "<h4",
                "<h5",
                "<h6",
                "<span",
                "<table",
                "<tr",
                "<td",
                "<th",
                "<ul",
                "<ol",
                "<header",
                "<footer",
                "<nav",
                # Head
                "<head",
                "<style",
                "<script",
                "<meta",
                "<title",
                "",
            ]
        elif language == Language.CSHARP:
            return [
                "\ninterface ",
                "\nenum ",
                "\nimplements ",
                "\ndelegate ",
                "\nevent ",
                # Split along class definitions
                "\nclass ",
                "\nabstract ",
                # Split along method definitions
                "\npublic ",
                "\nprotected ",
                "\nprivate ",
                "\nstatic ",
                "\nreturn ",
                # Split along control flow statements
                "\nif ",
                "\ncontinue ",
                "\nfor ",
                "\nforeach ",
                "\nwhile ",
                "\nswitch ",
                "\nbreak ",
                "\ncase ",
                "\nelse ",
                # Split by exceptions
                "\ntry ",
                "\nthrow ",
                "\nfinally ",
                "\ncatch ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.SOL:
            return [
                # Split along compiler information definitions
                "\npragma ",
                "\nusing ",
                # Split along contract definitions
                "\ncontract ",
                "\ninterface ",
                "\nlibrary ",
                # Split along method definitions
                "\nconstructor ",
                "\ntype ",
                "\nfunction ",
                "\nevent ",
                "\nmodifier ",
                "\nerror ",
                "\nstruct ",
                "\nenum ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\ndo while ",
                "\nassembly ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.COBOL:
            return [
                # Split along divisions
                "\nIDENTIFICATION DIVISION.",
                "\nENVIRONMENT DIVISION.",
                "\nDATA DIVISION.",
                "\nPROCEDURE DIVISION.",
                # Split along sections within DATA DIVISION
                "\nWORKING-STORAGE SECTION.",
                "\nLINKAGE SECTION.",
                "\nFILE SECTION.",
                # Split along sections within PROCEDURE DIVISION
                "\nINPUT-OUTPUT SECTION.",
                # Split along paragraphs and common statements
                "\nOPEN ",
                "\nCLOSE ",
                "\nREAD ",
                "\nWRITE ",
                "\nIF ",
                "\nELSE ",
                "\nMOVE ",
                "\nPERFORM ",
                "\nUNTIL ",
                "\nVARYING ",
                "\nACCEPT ",
                "\nDISPLAY ",
                "\nSTOP RUN.",
                # Split by the normal type of lines
                "\n",
                " ",
                "",
            ]
        elif language == Language.LUA:
            return [
                # Split along variable and table definitions
                "\nlocal ",
                # Split along function definitions
                "\nfunction ",
                # Split along control flow statements
                "\nif ",
                "\nfor ",
                "\nwhile ",
                "\nrepeat ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.HASKELL:
            return [
                # Split along function definitions
                "\nmain :: ",
                "\nmain = ",
                "\nlet ",
                "\nin ",
                "\ndo ",
                "\nwhere ",
                "\n:: ",
                "\n= ",
                # Split along type declarations
                "\ndata ",
                "\nnewtype ",
                "\ntype ",
                "\n:: ",
                # Split along module declarations
                "\nmodule ",
                # Split along import statements
                "\nimport ",
                "\nqualified ",
                "\nimport qualified ",
                # Split along typeclass declarations
                "\nclass ",
                "\ninstance ",
                # Split along case expressions
                "\ncase ",
                # Split along guards in function definitions
                "\n| ",
                # Split along record field declarations
                "\ndata ",
                "\n= {",
                "\n, ",
                # Split by the normal type of lines
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language == Language.POWERSHELL:
            return [
                # Split along function definitions
                "\nfunction ",
                # Split along parameter declarations (escape parentheses)
                "\nparam ",
                # Split along control flow statements
                "\nif ",
                "\nforeach ",
                "\nfor ",
                "\nwhile ",
                "\nswitch ",
                # Split along class definitions (for PowerShell 5.0 and above)
                "\nclass ",
                # Split along try-catch-finally blocks
                "\ntry ",
                "\ncatch ",
                "\nfinally ",
                # Split by normal lines and empty spaces
                "\n\n",
                "\n",
                " ",
                "",
            ]
        elif language in Language._value2member_map_:
            raise ValueError(f"Language {language} is not implemented yet!")
        else:
            raise ValueError(
                f"Language {language} is not supported! "
                f"Please choose from {list(Language)}"
            )


class spacyCutStep(TextSplitter):
    """Splitting text using Spacy package."""

    def __init__(self, nlp_language_model: str = "en_core_web_sm", **kwargs: Any) -> None:
        """Initialize the Spacy splitter."""
        super().__init__(**kwargs)
        try:
            import spacy
            #spacy.nlp.max_length = 50 * 1024 * 1024
        except ImportError:
            raise ImportError(
                "Spacy is not installed, please install it with `pip install spacy`."
            )
        self._nlp = spacy.load(nlp_language_model)
        self._nlp.max_length = 80 * 1024 * 1024

    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks."""
        doc = self._nlp(text)
        return [sent.text for sent in doc.sents]

class NLTKTextCutStep(TextSplitter):
    """Splitting text using NLTK package."""

    def __init__(
        self,
        language: str = "english",
        *,
        use_span_tokenize: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the NLTK splitter."""
        super().__init__(**kwargs)
        self._language = language
        self._use_span_tokenize = use_span_tokenize
        if self._use_span_tokenize and self._separator != "":
            raise ValueError("When use_span_tokenize is True, separator should be ''")
        try:
            if self._use_span_tokenize:
                from nltk.tokenize import _get_punkt_tokenizer

                self._tokenizer = _get_punkt_tokenizer(self._language)
            else:
                from nltk.tokenize import sent_tokenize

                self._tokenizer = sent_tokenize
        except ImportError:
            raise ImportError(
                "NLTK is not installed, please install it with `pip install nltk`."
            )

    def split_text(self, text: str) -> List[str]:
        """Split incoming text and return chunks."""
        # First we naively split the large input into a bunch of smaller ones.
        if self._use_span_tokenize:
            spans = list(self._tokenizer.span_tokenize(text))
            splits = []
            for i, (start, end) in enumerate(spans):
                if i > 0:
                    prev_end = spans[i - 1][1]
                    sentence = text[prev_end:start] + text[start:end]
                else:
                    sentence = text[start:end]
                splits.append(sentence)
        else:
            splits = self._tokenizer(text, language=self._language)
        return  splits  # self._merge_splits(splits, self._separator)


def combine_sentences(sentences: List[dict], buffer_size: int = 1) -> List[dict]:
    """Combine sentences based on buffer size.

    Args:
        sentences: List of sentences to combine.
        buffer_size: Number of sentences to combine. Defaults to 1.

    Returns:
        List of sentences with combined sentences.
    """

    # Go through each sentence dict
    for i in range(len(sentences)):
        # Create a string that will hold the sentences which are joined
        combined_sentence = ""

        # Add sentences before the current one, based on the buffer size.
        for j in range(i - buffer_size, i):
            # Check if the index j is not negative
            # (to avoid index out of range like on the first one)
            if j >= 0:
                # Add the sentence at index j to the combined_sentence string
                combined_sentence += sentences[j]["sentence"] + " "

        # Add the current sentence
        combined_sentence += sentences[i]["sentence"]

        # Add sentences after the current one, based on the buffer size
        for j in range(i + 1, i + 1 + buffer_size):
            # Check if the index j is within the range of the sentences list
            if j < len(sentences):
                # Add the sentence at index j to the combined_sentence string
                combined_sentence += " " + sentences[j]["sentence"]

        # Then add the whole thing to your dict
        # Store the combined sentence in the current sentence dict
        sentences[i]["combined_sentence"] = combined_sentence

    return sentences


def calculate_cosine_distances(sentences: List[dict]) -> Tuple[List[float], List[dict]]:
    """Calculate cosine distances between sentences.

    Args:
        sentences: List of sentences to calculate distances for.

    Returns:
        Tuple of distances and sentences.
    """
    distances = []
    for i in range(len(sentences) - 1):
        embedding_current = sentences[i]["combined_sentence_embedding"]
        embedding_next = sentences[i + 1]["combined_sentence_embedding"]

        # Calculate cosine similarity
        similarity = cosine_similarity([embedding_current], [embedding_next])[0][0]

        # Convert to cosine distance
        distance = 1 - similarity

        # Append cosine distance to the list
        distances.append(distance)

        # Store distance in the dictionary
        sentences[i]["distance_to_next"] = distance

    # Optionally handle the last sentence
    # sentences[-1]['distance_to_next'] = None  # or a default value

    return distances, sentences


BreakpointThresholdType = Literal[
    "percentile", "standard_deviation", "interquartile", "gradient"
]
BREAKPOINT_DEFAULTS: Dict[BreakpointThresholdType, float] = {
    "percentile": 95,
    "standard_deviation": 3,
    "interquartile": 1.5,
    "gradient": 95,
}


class GrammarSemanticSplitter(BaseDocumentTransformer):
    """Split the text based on semantic similarity.

    Taken from Greg Kamradt's wonderful notebook:
    https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb

    All credits to him.

    At a high level, this splits into sentences, then groups into groups of 3
    sentences, and then merges one that are similar in the embedding space.
    """

    def __init__(
        self,
        embeddings: Embeddings,
        buffer_size: int = 1,
        add_start_index: bool = False,
        breakpoint_threshold_type: BreakpointThresholdType = "percentile",
        breakpoint_threshold_amount: Optional[float] = None,
        number_of_chunks: Optional[int] = None,
        min_chunk_size: Optional[int] = None,
        max_chunk_size: Optional[int] = None,
        language: str = "english",
        use_span_tokenize: bool = False, # 是否保留分句后的空白符。
        
    ):
        self.max_chunk_size = max_chunk_size
        self._add_start_index = add_start_index
        self.embeddings = embeddings
        self.buffer_size = buffer_size
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.number_of_chunks = number_of_chunks
        # self.sentence_split_regex = sentence_split_regex
        # self.grammarCutStepSplitter = NLTKTextCutStep(language=language, use_span_tokenize=use_span_tokenize)
        self.grammarCutStepSplitter = spacyCutStep()
        if breakpoint_threshold_amount is None:
            self.breakpoint_threshold_amount = BREAKPOINT_DEFAULTS[
                breakpoint_threshold_type
            ]
        else:
            self.breakpoint_threshold_amount = breakpoint_threshold_amount
        self.min_chunk_size = min_chunk_size

    def _calculate_breakpoint_threshold(
        self, distances: List[float]
    ) -> Tuple[float, List[float]]:
        if self.breakpoint_threshold_type == "percentile":
            return cast(
                float,
                np.percentile(distances, self.breakpoint_threshold_amount),
            ), distances
        elif self.breakpoint_threshold_type == "standard_deviation":
            return cast(
                float,
                np.mean(distances)
                + self.breakpoint_threshold_amount * np.std(distances),
            ), distances
        elif self.breakpoint_threshold_type == "interquartile":
            q1, q3 = np.percentile(distances, [25, 75])
            iqr = q3 - q1

            return np.mean(
                distances
            ) + self.breakpoint_threshold_amount * iqr, distances
        elif self.breakpoint_threshold_type == "gradient":
            # Calculate the threshold based on the distribution of gradient of distance array. # noqa: E501
            distance_gradient = np.gradient(distances, range(0, len(distances)))
            return cast(
                float,
                np.percentile(distance_gradient, self.breakpoint_threshold_amount),
            ), distance_gradient
        else:
            raise ValueError(
                f"Got unexpected `breakpoint_threshold_type`: "
                f"{self.breakpoint_threshold_type}"
            )

    def _threshold_from_clusters(self, distances: List[float]) -> float:
        """
        Calculate the threshold based on the number of chunks.
        Inverse of percentile method.
        """
        if self.number_of_chunks is None:
            raise ValueError(
                "This should never be called if `number_of_chunks` is None."
            )
        x1, y1 = len(distances), 0.0
        x2, y2 = 1.0, 100.0

        x = max(min(self.number_of_chunks, x1), x2)

        # Linear interpolation formula
        if x2 == x1:
            y = y2
        else:
            y = y1 + ((y2 - y1) / (x2 - x1)) * (x - x1)

        y = min(max(y, 0), 100)

        return cast(float, np.percentile(distances, y))

    def _calculate_sentence_distances(
        self, single_sentences_list: List[str]
    ) -> Tuple[List[float], List[dict]]:
        """Split text into multiple components."""

        _sentences = [
            {"sentence": x, "index": i} for i, x in enumerate(single_sentences_list)
        ]
        sentences = combine_sentences(_sentences, self.buffer_size)
        embeddings = self.embeddings.embed_documents(
            [x["combined_sentence"] for x in sentences]
        )
        for i, sentence in enumerate(sentences):
            sentence["combined_sentence_embedding"] = embeddings[i]

        return calculate_cosine_distances(sentences)

    def split_text(
        self,
        text: str,
    ) -> List[str]:
        # Splitting the essay (by NLTK splitter)
        # single_sentences_list = re.split(self.sentence_split_regex, text)
        # mod to nltk splitter
        single_sentences_list = self.grammarCutStepSplitter.split_text(text)

        # having len(single_sentences_list) == 1 would cause the following
        # np.percentile to fail.
        if len(single_sentences_list) == 1:
            return single_sentences_list
        # similarly, the following np.gradient would fail
        if (
            self.breakpoint_threshold_type == "gradient"
            and len(single_sentences_list) == 2
        ):
            return single_sentences_list
        distances, sentences = self._calculate_sentence_distances(single_sentences_list)
        if self.number_of_chunks is not None:
            breakpoint_distance_threshold = self._threshold_from_clusters(distances)
            breakpoint_array = distances
        else:
            (
                breakpoint_distance_threshold,
                breakpoint_array,
            ) = self._calculate_breakpoint_threshold(distances)

        indices_above_thresh = [
            i
            for i, x in enumerate(breakpoint_array)
            if x > breakpoint_distance_threshold
        ]

        chunks = []
        start_index = 0

        # Iterate through the breakpoints to slice the sentences
        for index in indices_above_thresh:
            # The end index is the current breakpoint
            end_index = index

            # Slice the sentence_dicts from the current start index to the end index
            group = sentences[start_index : end_index + 1]
            combined_text = " ".join([d["sentence"] for d in group])
            token_cnt = len(Enc_token_cnt.encode(combined_text)) 

            # min_chunk_size控制
            # If specified, merge together small chunks.
            if (
                self.min_chunk_size is not None
                and token_cnt < self.min_chunk_size
            ):
                # len(combined_text) 改成算token_个数
                
                continue

            # max_chunk_size 控制
            if self.max_chunk_size is not None and token_cnt > self.max_chunk_size:
                # 需要进一步拆分
                temp_tokens = 0
                temp_group = []
                for d in group:
                    sent_tokens = len(Enc_token_cnt.encode(d["sentence"]))
                    if temp_tokens + sent_tokens > self.max_chunk_size:
                        if temp_group:
                            chunks.append(" ".join([x["sentence"] for x in temp_group]))
                        temp_group = [d]
                        temp_tokens = sent_tokens
                    else:
                        temp_group.append(d)
                        temp_tokens += sent_tokens
                if temp_group:
                    chunks.append(" ".join([x["sentence"] for x in temp_group]))
                start_index = end_index + 1
                continue


            # 正常合并语义相似度阈值之间的文本块
            # Update the start index for the next group
            chunks.append(combined_text)            
            start_index = index + 1

        # The last group, if any sentences remain
        if start_index < len(sentences):
            combined_text = " ".join([d["sentence"] for d in sentences[start_index:]])
            chunks.append(combined_text)
        return chunks

    def create_documents(
        self, texts: List[str], metadatas: Optional[List[dict]] = None
    ) -> List[Document]:
        """Create documents from a list of texts."""
        _metadatas = metadatas or [{}] * len(texts)
        documents = []
        for i, text in enumerate(texts):
            start_index = 0
            for chunk in self.split_text(text):
                metadata = copy.deepcopy(_metadatas[i])
                if self._add_start_index:
                    metadata["start_index"] = start_index
                new_doc = Document(page_content=chunk, metadata=metadata)
                documents.append(new_doc)
                start_index += len(chunk)
        return documents

    def split_documents(self, documents: Iterable[Document]) -> List[Document]:
        """Split documents."""
        texts, metadatas = [], []
        for doc in documents:
            texts.append(doc.page_content)
            metadatas.append(doc.metadata)
        return self.create_documents(texts, metadatas=metadatas)

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Transform sequence of documents by splitting them."""
        return self.split_documents(list(documents))
