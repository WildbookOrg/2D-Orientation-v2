from __future__ import absolute_import, division, print_function
from ibeis.control import controller_inject  # NOQA
import matplotlib.pyplot as plt
from os.path import abspath, expanduser, join, exists
import plottool as pt
import numpy as np
import utool as ut
import vtool as vt
import ibeis
import dtool
import time
import tqdm

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader

(print, rrr, profile) = ut.inject2(__name__)

_, register_ibs_method = controller_inject.make_ibs_register_decorator(__name__)
register_api = controller_inject.get_ibeis_flask_api(__name__)

register_preproc_annot = controller_inject.register_preprocs['annot']


ROOT = ibeis.const.ANNOTATION_TABLE


MAX_RANK = 12
FORCE_SERIAL = False
FORCE_SERIAL = FORCE_SERIAL or 'macosx' in ut.get_plat_specifier().lower()


DATA_DICT = {
    'mantaray'   : 'https://cthulhu.dyn.wildme.io/public/datasets/orientation.mantaray.coco.tar.gz',
    'rightwhale' : 'https://cthulhu.dyn.wildme.io/public/datasets/orientation.rightwhale.coco.tar.gz',
    'seadragon'  : 'https://cthulhu.dyn.wildme.io/public/datasets/orientation.seadragon.coco.tar.gz',
    'seaturtle'  : 'https://cthulhu.dyn.wildme.io/public/datasets/orientation.seaturtle.coco.tar.gz',
    'hammerhead' : 'https://cthulhu.dyn.wildme.io/public/datasets/orientation.hammerhead.coco.tar.gz',
}


URL_DICT = {
    'mantaray_v0'   : 'https://wildbookiarepository.azureedge.net/models/orientation.2d.manta.v0.pth',
    'rightwhale_v0' : 'https://wildbookiarepository.azureedge.net/models/orientation.2d.right_whale.v0.pth',
    'seadragon_v0'  : 'https://wildbookiarepository.azureedge.net/models/orientation.2d.sea_dragon.v0.pth',
    'seaturtle_v0'  : 'https://wildbookiarepository.azureedge.net/models/orientation.2d.sea_turtle.v0.pth',

    'mantaray_v1'   : 'https://wildbookiarepository.azureedge.net/models/orientation.2d.manta.v1.pth',
    'rightwhale_v1' : 'https://wildbookiarepository.azureedge.net/models/orientation.2d.right_whale.v1.pth',
    'seadragon_v1'  : 'https://wildbookiarepository.azureedge.net/models/orientation.2d.sea_dragon.v1.pth',
    'seaturtle_v1'  : 'https://wildbookiarepository.azureedge.net/models/orientation.2d.sea_turtle.v1.pth',
}


SPECIES_MODEL_TAG_MAPPING = {
    'manta_ray_giant'       : 'mantaray_v1',
    'right_whale_head'      : 'rightwhale_v1',
    'seadragon_weedy+head'  : 'seadragon_v1',
    'turtle_hawksbill+head' : 'seaturtle_v1',
}


def rank(ibs, result):
    cm_dict = result['cm_dict']
    cm_key = list(cm_dict.keys())[0]
    cm = cm_dict[cm_key]

    query_name = cm['qname']
    qnid = ibs.get_name_rowids_from_text(query_name)

    annot_uuid_list = cm['dannot_uuid_list']
    annot_score_list = cm['annot_score_list']
    daid_list = ibs.get_annot_aids_from_uuid(annot_uuid_list)
    dnid_list = ibs.get_annot_nids(daid_list)
    dscore_list = sorted(zip(annot_score_list, dnid_list), reverse=True)

    annot_ranks = []
    for rank, (dscore, dnid) in enumerate(dscore_list):
        if dnid == qnid:
            annot_ranks.append(rank)

    name_list = cm['unique_name_list']
    name_score_list = cm['name_score_list']
    dnid_list = ibs.get_name_rowids_from_text(name_list)
    dscore_list = sorted(zip(name_score_list, dnid_list), reverse=True)

    name_ranks = []
    for rank, (dscore, dnid) in enumerate(dscore_list):
        if dnid == qnid:
            name_ranks.append(rank)

    return annot_ranks, name_ranks


def rank_min_avg(rank_dict, max_rank):
    min_x_list, min_y_list = [], []
    avg_x_list, avg_y_list = [], []
    for rank in range(max_rank):
        count_min, count_avg, total = 0.0, 0.0, 0.0
        for qaid in rank_dict:
            annot_ranks = rank_dict[qaid]
            if len(annot_ranks) > 0:
                annot_min_rank = min(annot_ranks)
                annot_avg_rank = sum(annot_ranks) / len(annot_ranks)
                if annot_min_rank <= rank:
                    count_min += 1
                if annot_avg_rank <= rank:
                    count_avg += 1
            total += 1
        percentage_min = count_min / total
        min_x_list.append(rank + 1)
        min_y_list.append(percentage_min)
        percentage_avg = count_avg / total
        avg_x_list.append(rank + 1)
        avg_y_list.append(percentage_avg)

    min_vals = min_x_list, min_y_list
    avg_vals = avg_x_list, avg_y_list

    return min_vals, avg_vals


def get_marker(index, total):
    marker_list = ['o', 'X', '+', '*']
    num_markers = len(marker_list)
    if total <= 12:
        index_ = 0
    else:
        index_ = index % num_markers
    marker = marker_list[index_]
    return marker


@register_ibs_method
@register_api('/api/plugin/orientation/2d/', methods=['GET'])
def ibeis_plugin_orientation_2d_inference(ibs, aid_list, model_tag, device=None,
                                          batch_size=8, multi=True):
    r"""
    Run inference with 2D orientation estimation, as developed by Henry Grover

    Args:
        ibs       (IBEISController): IBEIS controller object
        aid_list  (list of int): A list of IBEIS Annotation IDs (aids)
        model_tag (string): Key to URL_DICT entry for this model

    Returns:
        degree_list

    CommandLine:
        python -m ibeis_2d_orientation._plugin --test-ibeis_plugin_orientation_2d_inference

    Example0:
        >>> # ENABLE_DOCTEST
        >>> import ibeis
        >>> import random
        >>> import utool as ut
        >>> from ibeis.init import sysres
        >>> import numpy as np
        >>> dbdir = sysres.ensure_testdb_orientation()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_valid_aids()
        >>> note_list = ibs.get_annot_notes(aid_list)
        >>> species_list = ibs.get_annot_species(aid_list)
        >>> flag_list = [
        >>>     note == 'random-00' and species == 'manta_ray_giant'
        >>>     for note, species in zip(note_list, species_list)
        >>> ]
        >>> aid_list = ut.compress(aid_list, flag_list)
        >>> random.seed(1)
        >>> random.shuffle(aid_list)
        >>> aid_list = aid_list[:5]
        >>> model_tag = 'mantaray_v1'
        >>> degree_list = ibs.ibeis_plugin_orientation_2d_inference(aid_list, model_tag)
        >>> result = degree_list
        >>> print(result)
        [101.2, 145.93, 86.38, 112.2, 60.96]
    """
    from ibeis_2d_orientation.turtles_test.train import separate_trig_to_angle
    from ibeis_2d_orientation.turtles_test.data import Data_turtles

    # Configuration values for model
    nClasses = 2
    separate_trig = True

    # Canonicalize the device with PyTorch and load GPU (cuda0) by default
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda:0'
        else:
            device = 'cpu'
    device = torch.device(device)

    # Download the model from the Azure CDN, if model_tag is recognized
    model_url = URL_DICT.get(model_tag, None)
    message = 'The specified model_tag = %r was not recognized' % (model_tag, )
    assert model_url is not None, message
    weight_filepath = ut.grab_file_url(model_url, appname='ibeis_2d_orientation',
                                       check_hash=True)

    # Create model instance, add ending classifier with number of classes = 2
    #
    # We set pre-trained to False here because we will be replacing them
    # with pre-trained values, no need to download the model from PyTorch model
    # zoo.
    model = torchvision.models.densenet161(pretrained=False)
    model.classifier = nn.Linear(2208, nClasses)

    # Load model from disk into memory and update model's weights
    if device.type in ['cpu']:
        print('Running inference on CPU')
        weight_data = torch.load(weight_filepath, map_location=device)
        using_gpu = False
    else:
        print('Running inference on GPU: %r' % (device, ))
        weight_data = torch.load(weight_filepath)
        using_gpu = True

        # Make parallel at end
        if multi:
            print('Using Multiple GPUs')
            model = nn.DataParallel(model)

    # Load weights and send to device
    model.load_state_dict(weight_data)
    model = model.to(device)

    # Pre-execute model state
    model.eval()

    ibs._parallel_chips = not FORCE_SERIAL
    filepath_list = ibs.get_annot_chip_fpath(aid_list, ensure=True)

    class DummyArgs(object):
        def __init__(self, filepath_list):
            self.filename_list = filepath_list
            self.filename_test = True
            self.degree_loss   = True
            self.show          = False

    args = DummyArgs(filepath_list)
    dataset = Data_turtles(dataType='test2020', experiment_type='test', args=args,
                           add_pad=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            drop_last=False, num_workers=0, pin_memory=using_gpu)

    start = time.time()

    with torch.no_grad():
        counter = 0
        outputs = []
        for inputs in tqdm.tqdm(dataloader, desc='test'):
            print('Loading batch %d from disk' % (counter, ))
            inputs = inputs.to(device)
            print(inputs.shape)
            print('Moving batch %d to device' % (counter, ))
            with torch.set_grad_enabled(False):
                print('Pre-model inference %d' % (counter, ))
                output = model(inputs)
                if separate_trig:
                    output = separate_trig_to_angle(output, args=args)
                print('Post-model inference %d' % (counter, ))
                outputs += output.tolist()
                print('Outputs done %d' % (counter, ))
            counter += 1

    time_elapsed = time.time() - start
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    result_list = []
    for output in outputs:
        result = output % 360
        result = round(result, 2)
        result_list.append(result)

    return result_list


class Orientation2DConfig(dtool.Config):
    def get_param_info_list(self):
        return [
            ut.ParamInfo('orientation_2d_model_tag', None, valid_values=list(URL_DICT.keys())),
        ]


@register_preproc_annot(
    tablename='orientation_two_dimension', parents=[ROOT],
    colnames=['degree'],
    coltypes=[float],
    configclass=Orientation2DConfig,
    fname='orientation',
    chunksize=128,
)
def ibeis_plugin_orientation_2d_inference_depc(depc, aid_list, config=None):
    r"""
    Run inference with 2D orientation estimation with dependency cache (depc)

    Args:
        depc      (Dependency Cache): IBEIS dependency cache object
        aid_list  (list of int): list of annot rowids (aids)
        config    (Orientation2DConfig): config for depcache

    CommandLine:
        python -m ibeis_2d_orientation._plugin --test-ibeis_plugin_orientation_2d_inference_depc

    Example0:
        >>> # ENABLE_DOCTEST
        >>> import ibeis
        >>> import random
        >>> import utool as ut
        >>> from ibeis.init import sysres
        >>> import numpy as np
        >>> dbdir = sysres.ensure_testdb_orientation()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> aid_list = ibs.get_valid_aids()
        >>> note_list = ibs.get_annot_notes(aid_list)
        >>> species_list = ibs.get_annot_species(aid_list)
        >>> flag_list = [
        >>>     note == 'random-00' and species == 'manta_ray_giant'
        >>>     for note, species in zip(note_list, species_list)
        >>> ]
        >>> aid_list = ut.compress(aid_list, flag_list)
        >>> random.seed(1)
        >>> random.shuffle(aid_list)
        >>> aid_list = aid_list[:5]
        >>> config = {
        >>>     'orientation_2d_model_tag': 'mantaray_v1',
        >>> }
        >>> degree_list = ibs.depc_annot.get('orientation_two_dimension', aid_list, 'degree', config=config)
        >>> result = degree_list
        >>> print(result)
        [101.2, 145.93, 86.38, 112.2, 60.96]
    """
    ibs = depc.controller

    model_tag = config['orientation_2d_model_tag']
    assert model_tag is not None

    values = ibs.ibeis_plugin_orientation_2d_inference(aid_list, model_tag)

    for degree in values:
        yield (
            degree,
        )


@register_ibs_method
def ibeis_plugin_orientation_2d_render_examples(ibs, num_examples=10, use_depc=True,
                                                desired_note='random-00', **kwargs):
    r"""
    Show examples of the prediction for each species

    Args:
        ibs       (IBEISController): IBEIS controller object
        aid_list  (list of int): A list of IBEIS Annotation IDs (aids)
        model_tag (string): Key to URL_DICT entry for this model

    Returns:
        theta_list

    CommandLine:
        python -m ibeis_2d_orientation._plugin --test-ibeis_plugin_orientation_2d_render_examples

    Example0:
        >>> # ENABLE_DOCTEST
        >>> import ibeis
        >>> import random
        >>> from ibeis.init import sysres
        >>> import numpy as np
        >>> dbdir = sysres.ensure_testdb_orientation()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> fig_filepath = ibs.ibeis_plugin_orientation_2d_render_examples()
        >>> print(fig_filepath)
    """
    import random
    random.seed(1)

    aid_list = ibs.get_valid_aids()
    note_list = ibs.get_annot_notes(aid_list)
    note_list = np.array(note_list)
    flag_list = note_list == desired_note

    aid_list = ut.compress(aid_list, flag_list)
    species_list = ibs.get_annot_species(aid_list)
    species_list = np.array(species_list)

    result_dict = {}
    all_aid_list_ = []
    key_list = sorted(list(SPECIES_MODEL_TAG_MAPPING.keys()))
    for key in key_list:
        model_tag = SPECIES_MODEL_TAG_MAPPING.get(key, None)
        assert model_tag is not None
        flag_list = species_list == key
        aid_list_ = ut.compress(aid_list, flag_list)
        random.shuffle(aid_list_)
        aid_list_ = aid_list_[:num_examples]
        if use_depc:
            config = {
                'orientation_2d_model_tag': model_tag,
            }
            result_list = ibs.depc_annot.get('orientation_two_dimension', aid_list_,
                                             'degree', config=config)
        else:
            result_list = ibs.ibeis_plugin_orientation_2d_inference(aid_list_,
                                                                    model_tag,
                                                                    **kwargs)
        result_dict[key] = list(zip(aid_list_, result_list))
        all_aid_list_ += aid_list_

    key_list = list(result_dict.keys())

    slots = (len(key_list), num_examples, )
    figsize = (10 * slots[1], 10 * slots[0], )
    fig_ = plt.figure(figsize=figsize, dpi=150)  # NOQA
    plt.grid(None)

    config2_ = {
        'resize_dim'   : 'wh',
        'dim_size'     : (256, 256),
        # 'axis_aligned' : True,
    }
    # Pre-compute in parallel quickly so they are cached
    ibs.get_annot_chip_fpath(all_aid_list_, ensure=True, config2_=config2_)

    index = 1
    key_list = sorted(key_list)
    for row, key in enumerate(key_list):
        value_list  = result_dict[key]
        for col, value in enumerate(value_list):
            aid, degree = value

            axes_ = plt.subplot(slots[0], slots[1], index)
            axes_.set_title('Degree = %0.02f' % (degree, ))
            axes_.axis('off')

            chip = ibs.get_annot_chips(aid, config2_=config2_)
            chip = chip[:, :, ::-1]

            chip_ = chip.copy()
            theta = ut.deg_to_rad(degree)
            chip_ = vt.rotate_image(chip_, theta)

            canvas = np.hstack((chip, chip_))
            plt.imshow(canvas)
            index += 1

    fig_filename = 'orientation.2d.examples.predictions.png'
    fig_path = abspath(expanduser(join('~', 'Desktop')))
    fig_filepath = join(fig_path, fig_filename)
    plt.savefig(fig_filepath, bbox_inches='tight')

    return fig_filepath


@register_ibs_method
def ibeis_plugin_orientation_2d_render_feasability(ibs, desired_species, desired_notes=None,
                                                   use_depc=True, **kwargs):
    r"""
    Show examples of the prediction for each species

    Args:
        ibs       (IBEISController): IBEIS controller object
        aid_list  (list of int): A list of IBEIS Annotation IDs (aids)
        model_tag (string): Key to URL_DICT entry for this model

    Returns:
        theta_list

    CommandLine:
        python -m ibeis_2d_orientation._plugin --test-ibeis_plugin_orientation_2d_render_feasability

    Example0:
        >>> # ENABLE_DOCTEST
        >>> import ibeis
        >>> import random
        >>> from ibeis.init import sysres
        >>> import numpy as np
        >>> dbdir = sysres.ensure_testdb_orientation()
        >>> ibs = ibeis.opendb(dbdir=dbdir)
        >>> species_list = [
        >>>     'right_whale_head',
        >>>     'manta_ray_giant',
        >>>     'seadragon_weedy+head',
        >>>     'turtle_hawksbill+head'
        >>> ]
        >>> for species in species_list:
        >>>     fig_filepath = ibs.ibeis_plugin_orientation_2d_render_feasability(species)
        >>>     print(fig_filepath)
    """
    if desired_notes is None:
        desired_notes = [
            'source',
            'aligned',
            'random-00',
            'random-01',
            'random-02',
            'source*',
            'aligned*',
            'random-00*',
            'random-01*',
            'random-02*',
        ]

    # Load any pre-computed ranks
    rank_dict_filepath = join(ibs.dbdir, 'ranks.%s.pkl' % (desired_species, ))
    print('Using cached rank file: %r' % (rank_dict_filepath, ))
    if exists(rank_dict_filepath):
        rank_dict = ut.load_cPkl(rank_dict_filepath)
    else:
        rank_dict = {}

    query_config_dict_dict = {
        'HS' : {},
    }

    aid_dict = {}

    for desired_note in desired_notes:
        aid_list = ibs.get_valid_aids()

        note_list = ibs.get_annot_notes(aid_list)
        note_list = np.array(note_list)
        flag_list = note_list == desired_note.strip('*')
        aid_list = ut.compress(aid_list, flag_list)

        species_list = ibs.get_annot_species(aid_list)
        species_list = np.array(species_list)
        flag_list = species_list == desired_species
        aid_list = ut.compress(aid_list, flag_list)

        if desired_note.endswith('*'):

            model_tag = SPECIES_MODEL_TAG_MAPPING.get(desired_species, None)
            assert model_tag is not None
            config = {
                'orientation_2d_model_tag': model_tag,
            }
            degree_list = ibs.depc_annot.get('orientation_two_dimension', aid_list,
                                             'degree', config=config)

            all_aid_list = ibs.get_valid_aids()
            existing_species_list = ibs.get_annot_species(all_aid_list)
            existing_species_list = np.array(existing_species_list)
            existing_note_list = ibs.get_annot_notes(all_aid_list)
            existing_note_list = np.array(existing_note_list)
            delete_species_flag_list = existing_species_list == desired_species
            delete_note_flag_list = existing_note_list == desired_note
            delete_flag_list = delete_species_flag_list & delete_note_flag_list
            delete_aid_list = ut.compress(all_aid_list, delete_flag_list)

            gid_list       = ibs.get_annot_gids(aid_list)
            bbox_list      = ibs.get_annot_bboxes(aid_list)
            theta_list     = ibs.get_annot_thetas(aid_list)
            species_list   = ibs.get_annot_species(aid_list)
            viewpoint_list = ibs.get_annot_viewpoints(aid_list)
            name_list      = ibs.get_annot_names(aid_list)
            note_list      = [desired_note] * len(aid_list)

            theta_list_ = []
            for theta, degree in zip(theta_list, degree_list):
                theta_ = ut.deg_to_rad(degree)
                theta_ = (theta - theta_) % (2.0 * np.pi)
                theta_list_.append(theta_)

            aid_list_ = ibs.add_annots(
                gid_list,
                bbox_list=bbox_list,
                theta_list=theta_list_,
                species_list=species_list,
                viewpoint_list=viewpoint_list,
                name_list=name_list,
                notes_list=note_list
            )

            delete_aid_list = list(set(delete_aid_list) - set(aid_list_))
            ibs.delete_annots(delete_aid_list)

            aid_list = aid_list_

        nid_list = ibs.get_annot_nids(aid_list)
        assert sum(np.array(nid_list) <= 0) == 0

        args = (len(aid_list), len(set(nid_list)), desired_species, desired_note, )
        print('Using %d annotations of %d names for species %r (note = %r)' % args)
        print('\t Species    : %r' % (set(ibs.get_annot_species(aid_list)), ))
        print('\t Viewpoints : %r' % (set(ibs.get_annot_viewpoints(aid_list)), ))

        if len(aid_list) == 0:
            print('\tSKIPPING')
            continue

        for qindex, qaid in tqdm.tqdm(list(enumerate(aid_list))):
            n = 1 if qindex <= 20 else 0

            qaid_list = [qaid]
            daid_list = aid_list

            print('Processing AID %d' % (qaid, ))
            for query_config_label in query_config_dict_dict:
                query_config_dict = query_config_dict_dict[query_config_label]

                # label = query_config_label
                label = '%s %s' % (query_config_label, desired_note, )

                if label not in aid_dict:
                    aid_dict[label] = []
                aid_dict[label].append(qaid)

                if label not in rank_dict:
                    rank_dict[label] = {
                        'annots': {},
                        'names': {},
                    }

                flag1 = qaid not in rank_dict[label]['annots']
                flag2 = qaid not in rank_dict[label]['names']
                if flag1 or flag2:
                    query_result = ibs.query_chips_graph(
                        qaid_list=qaid_list,
                        daid_list=daid_list,
                        query_config_dict=query_config_dict,
                        echo_query_params=False,
                        cache_images=True,
                        n=n,
                    )
                    annot_ranks, name_ranks = rank(ibs, query_result)
                    rank_dict[label]['annots'][qaid] = annot_ranks
                    rank_dict[label]['names'][qaid] = name_ranks

            # if qindex % 10 == 0:
            #     ut.save_cPkl(rank_dict_filepath, rank_dict)

        ut.save_cPkl(rank_dict_filepath, rank_dict)

    #####

    rank_dict_ = {}
    for label in rank_dict:
        annot_ranks = rank_dict[label]['annots']
        name_ranks = rank_dict[label]['names']

        annot_ranks_ = {}
        for qaid in annot_ranks:
            if qaid in aid_dict[label]:
                annot_ranks_[qaid] = annot_ranks[qaid]

        name_ranks_ = {}
        for qaid in name_ranks:
            if qaid in aid_dict[label]:
                name_ranks_[qaid] = name_ranks[qaid]

        rank_dict_[label] = {
            'annots' : annot_ranks_,
            'names'  : name_ranks_,
        }

    fig_ = plt.figure(figsize=(20, 10), dpi=300)  # NOQA

    rank_label_list = list(rank_dict_.keys())

    source_list, original_list, matched_list, unmatched_list = [], [], [], []
    label_list = []
    for desired_note in desired_notes:
        for query_config_label in query_config_dict_dict:
            label = '%s %s' % (query_config_label, desired_note, )
            if label not in rank_label_list:
                continue
            if desired_note == 'source':
                source_list.append(label)
            elif desired_note.endswith('*'):
                label_ = label.strip('*')
                if label_ in rank_label_list:
                    matched_list.append(label)
                else:
                    unmatched_list.append(label)
            else:
                original_list.append(label)
            label_list.append(label)

    assert len(source_list) <= 1
    color_label_list = original_list + unmatched_list
    color_list = pt.distinct_colors(len(color_label_list), randomize=False)

    color_dict = {}
    line_dict = {}

    for label in source_list:
        color_dict[label] = (0.0, 0.0, 0.0)
        line_dict[label] = '-'

    for label, color in zip(color_label_list, color_list):
        color_dict[label] = color
        if label in unmatched_list:
            line_dict[label] = '--'
        else:
            line_dict[label] = '-'

    for label in matched_list:
        label_ = label.strip('*')
        color = color_dict.get(label_, None)
        assert color is not None
        color_dict[label] = color
        line_dict[label] = '--'

    color_list = ut.take(color_dict, label_list)
    line_list = ut.take(line_dict, label_list)
    assert None not in color_list and None not in line_list

    values_list = []
    for label in label_list:
        annot_ranks = rank_dict_[label]['annots']
        min_vals,   avg_vals   = rank_min_avg(annot_ranks, MAX_RANK)
        min_x_list, min_y_list = min_vals
        # avg_x_list, avg_y_list = avg_vals
        values_list.append(
            (
                label,
                min_x_list,
                min_y_list,
            )
        )

    axes_ = plt.subplot(121)
    axes_.set_autoscalex_on(False)
    axes_.set_autoscaley_on(False)
    axes_.set_ylabel('Percentage')
    axes_.set_xlabel('Rank')
    axes_.set_xlim([1.0, MAX_RANK])
    axes_.set_ylim([0.0, 1.0])
    zipped = list(zip(color_list, line_list, values_list))
    total = len(zipped)
    for index, (color, linestyle, values) in enumerate(zipped):
        label, x_list, y_list = values
        marker = get_marker(index, total)
        plt.plot(x_list, y_list, color=color, marker=marker, label=label,
                 linestyle=linestyle, alpha=1.0)

    plt.title('One-to-Many Annotations - Cumulative Match Rank')
    plt.legend(bbox_to_anchor=(0.0, 1.04, 1.0, .102), loc=3, ncol=2, mode="expand",
               borderaxespad=0.0)

    values_list = []
    for label in label_list:
        name_ranks = rank_dict_[label]['names']
        min_vals,   avg_vals   = rank_min_avg(name_ranks, MAX_RANK)
        min_x_list, min_y_list = min_vals
        values_list.append(
            (
                label,
                min_x_list,
                min_y_list,
            )
        )

    axes_ = plt.subplot(122)
    axes_.set_autoscalex_on(False)
    axes_.set_autoscaley_on(False)
    axes_.set_ylabel('Percentage')
    axes_.set_xlabel('Rank')
    axes_.set_xlim([1.0, MAX_RANK])
    axes_.set_ylim([0.0, 1.0])
    zipped = list(zip(color_list, line_list, values_list))
    total = len(zipped)
    for index, (color, linestyle, values) in enumerate(zipped):
        label, x_list, y_list = values
        marker = get_marker(index, total)
        plt.plot(x_list, y_list, color=color, marker=marker, label=label,
                 linestyle=linestyle, alpha=1.0)

    plt.title('One-to-Many Names - Cumulative Match Rank')
    plt.legend(bbox_to_anchor=(0.0, 1.04, 1.0, .102), loc=3, ncol=2, mode="expand",
               borderaxespad=0.0)

    note_str = '_'.join(desired_notes)
    args = (desired_species, note_str, )
    fig_filename = 'orientation.2d.matching.hotspotter.%s.%s.png' % args
    fig_path = abspath(expanduser(join('~', 'Desktop')))
    fig_filepath = join(fig_path, fig_filename)
    plt.savefig(fig_filepath, bbox_inches='tight')

    return fig_filepath

if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis_2d_orientation._plugin --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
