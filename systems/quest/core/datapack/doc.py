from typing import Any, Dict, Optional

class Doc:
    def __init__(self, content: Any, doc_id: int, metadata: Optional[Dict[str, str]] = None):
        self.content = content
        self.doc_id = doc_id
        self.metadata = metadata if metadata is not None else {}

    def __repr__(self):
        doc_content = str(self.content)[0:30] + "... "
        doc_str = doc_content + f"Doc(doc_id={self.doc_id}, metadata={self.metadata})"
        return doc_str

class ZenDBDoc(Doc):
    attr_dict: dict[str, Any]

    def __init__(self, attr_dict):
        super().__init__(content = None, doc_id = attr_dict["doc_id"])
        self.metadata = {"file_name" :attr_dict["name"]}
        attr_dict["file_type"] = "md"
        self.attr_dict = attr_dict

    def _get_attr(self, attr: str) -> Any:
        return self.attr_dict[attr]

    def __getitem__(self, key):
        return self.attr_dict[key]
    
    def __setitem__(self, key, value):
        self.attr_dict[key] = value
    
    def __delitem__(self, key):
        del self.attr_dict[key]
    
    def __len__(self):
        return len(self.attr_dict)
    
    def __iter__(self):
        return iter(self.attr_dict)
    
    def __contains__(self, key):
        return key in self.attr_dict
    
    def __repr__(self):
        return repr(self.attr_dict)
    
    def get(self, key, default=None):
        return self.attr_dict.get(key, default)
    
    def keys(self):
        return self.attr_dict.keys()
    
    def values(self):
        return self.attr_dict.values()
    
    def items(self):
        return self.attr_dict.items()
    



class TextDoc(Doc):
    """
    Represents a document with its content and metadata.
    """

    def __init__(self, content: str, doc_id: int, metadata: Optional[Dict[str, str]] = None):
        self.content = content
        self.doc_id = doc_id
        self.metadata = metadata if metadata is not None else {}

    def __repr__(self):
        return f"Doc(content={self.content[:30]}..., doc_id={self.doc_id}, metadata={self.metadata})"