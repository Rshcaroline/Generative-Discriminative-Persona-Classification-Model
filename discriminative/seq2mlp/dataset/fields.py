import logging

import torchtext
from torchtext import data
import nltk

class SourceField(torchtext.data.Field):
    """ Wrapper class of torchtext.data.Field that forces batch_first and include_lengths to be True. """

    def __init__(self, **kwargs):
        logger = logging.getLogger(__name__)

        if kwargs.get('batch_first') is False:
            logger.warning("Option batch_first has to be set to use pytorch-seq2mlp.  Changed to True.")
        kwargs['batch_first'] = True
        if kwargs.get('include_lengths') is False:
            logger.warning("Option include_lengths has to be set to use pytorch-seq2mlp.  Changed to True.")
        kwargs['include_lengths'] = True
        # kwargs['lower'] = True
        # kwargs['tokenize'] = data.get_tokenizer('moses')

        super(SourceField, self).__init__(**kwargs)

class TargetField(torchtext.data.Field):
    """ Wrapper class of torchtext.data.Field that forces batch_first to be True and prepend <sos> and append <eos> to sequences in preprocessing step.

    Attributes:
        sos_id: index of the start of sentence symbol
        eos_id: index of the end of sentence symbol
    """

    SYM_SOS = '<sos>'
    SYM_EOS = '<eos>'

    def __init__(self, **kwargs):
        logger = logging.getLogger(__name__)

        if kwargs.get('batch_first') == False:
            logger.warning("Option batch_first has to be set to use pytorch-seq2mlp.  Changed to True.")
        kwargs['batch_first'] = True
        if kwargs.get('preprocessing') is None:
            kwargs['preprocessing'] = lambda seq: [self.SYM_SOS] + seq + [self.SYM_EOS]
        else:
            func = kwargs['preprocessing']
            kwargs['preprocessing'] = lambda seq: [self.SYM_SOS] + func(seq) + [self.SYM_EOS]
        kwargs['include_lengths'] = True
        # kwargs['tokenize'] = data.get_tokenizer('moses')
        # kwargs['lower'] = True

        self.sos_id = None
        self.eos_id = None

        super(TargetField, self).__init__(**kwargs)

    def build_vocab(self, *args, **kwargs):
        super(TargetField, self).build_vocab(*args, **kwargs)
        self.sos_id = self.vocab.stoi[self.SYM_SOS]
        self.eos_id = self.vocab.stoi[self.SYM_EOS]

class SpeakerField(torchtext.data.Field):
    """ Wrapper class of torchtext.data.Field that pad and unknow token are None. """

    def __init__(self, **kwargs):
        logger = logging.getLogger(__name__)

        kwargs['pad_token'] = None
        kwargs['unk_token'] = None

        super(SpeakerField, self).__init__(**kwargs)
