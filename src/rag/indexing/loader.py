import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document

_LILIAN_CLASSES = ("post-content", "post-title", "post-header")


def load_web(url: str, css_classes: list[str] | None = None) -> list[Document]:
    """Load a web page, optionally restricting to specific CSS classes."""
    classes = tuple(css_classes) if css_classes else _LILIAN_CLASSES
    loader = WebBaseLoader(
        web_paths=(url,),
        bs_kwargs={"parse_only": bs4.SoupStrainer(class_=classes)},
    )
    return loader.load()


def load_youtube(video_url: str) -> list[Document]:
    """Load a YouTube video transcript as a Document."""
    from langchain_community.document_loaders import YoutubeLoader

    return YoutubeLoader.from_youtube_url(video_url, add_video_info=True).load()
