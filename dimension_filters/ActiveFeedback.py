import numpy as np
from .AbstractFilter import AbstractFilter


class ActiveFeedback(AbstractFilter):
    """
    """

    def __init__(self, qrys_encoder, **kwargs):
        super().__init__(qrys_encoder, **kwargs)

        self.docs_encoder = kwargs["docs_encoder"]
        self.run = kwargs["run"]

    def _single_importance(self, query):
        qemb = self.qrys_encoder.get_encoding(query.query_id)
        dlist = self.run[(self.run.query_id == query.query_id)].sort_values(by='relevance', ascending=False).doc_id.to_list()
        posemb = self.docs_encoder.get_encoding(dlist[0])
        itx_vec = np.multiply(qemb, posemb)
        return itx_vec