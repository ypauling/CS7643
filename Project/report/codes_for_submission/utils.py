import os
import copy
import simplejson as json


def fullpath(ext, ROOT):
    return os.path.join(ROOT, ext)


class Layer(object):

    L1 = 'layer1'
    L2 = 'layer2'
    L3 = 'layer3'
    INGRS = 'det_ingrs'
    GOODIES = 'goodies'

    @staticmethod
    def load(name, ROOT):
        with open(fullpath(name + '.json', ROOT)) as f:
            return json.load(f)

    @staticmethod
    def merge(layers, ROOT, copy_base=False):

        layers = [layer if isinstance(layer, list) else Layer.load(
            layer, ROOT) for layer in layers]
        base = copy.deepcopy(layers[0]) if copy_base else layers[0]

        entries_by_id = {entry['id']: entry for entry in base}
        for layer in layers[1:]:
            for entry in layer:
                base_entry = entries_by_id.get(entry['id'])
                if not base_entry:
                    continue
                base_entry.update(entry)

        return base
