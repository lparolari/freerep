import torch
from torchvision.tv_tensors import Video, BoundingBoxes
from torchvision.transforms.v2.functional import to_image


from polypsense.dataset.repo.fragmenter import Fragmenter
from polypsense.dataset.repo.identifier import Identifier
from polypsense.dataset.repo.videoer import Videoer


class FragmentIdentityDataset:
    """
    A dataset of pairs (fragment, identity).
    """

    def __init__(
        self,
        instance_ds,
        fragmenter,
        identifier,
        videoer,
        fragment_transforms=None,
        return_dict=False,
    ):
        self.instance_ds = instance_ds
        self.fragmenter: Fragmenter = fragmenter
        self.identifier: Identifier = identifier
        self.videoer: Videoer = videoer
        self.fragment_transforms = fragment_transforms
        self.return_dict = return_dict

        self.ids: list[int] = None
        self.x: list = None
        self.y: list = None
        self._build()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        fragment, identity, video = self.x[id], self.y[id], self.v[id]

        clip = self._get_frames(fragment)  # [s, c, w, h]
        bboxes = self._get_bboxes(fragment)  # [s, 4]
        position_identity = self._get_position_realtive_to_identity(fragment)  # [1]
        position_video = self._get_position_realtive_to_video(fragment)  # [1]
        category_id = self._get_category_id(fragment)  # [1]
        size = self._get_size(fragment)
        histology = self._get_histology(fragment)
        has_lesion = self._has_lesion(fragment)

        if self.fragment_transforms:
            clip, bboxes = self.fragment_transforms(clip, bboxes)

        if self.return_dict:
            return {
                "clip": clip,
                "bboxes": bboxes,
                "label": identity,  # legacy: required in e2e.model
                "identity": identity,
                "video": video,
                "position": position_identity,  # legacy: this is required by e2e
                "position_identity": position_identity,
                "position_video": position_video,
                "category_id": category_id,
                "size": size,
                "histology": histology,
                "has_lesion": has_lesion,
            }

        return clip, identity  # [s, c, w, h], [1]

    def _get_frames(self, fragment):
        frames = [self.instance_ds.img(frame_id) for frame_id in fragment]
        frames = [to_image(frame) for frame in frames]
        return Video(torch.stack(frames))

    def _get_bboxes(self, fragment):
        img_ann_0 = self.instance_ds.img_ann(fragment[0])

        format = "xywh"
        canvas_size = (img_ann_0["height"], img_ann_0["width"])

        bboxes = [self.instance_ds.tgt_ann(frame_id)["bbox"] for frame_id in fragment]
        bboxes = torch.tensor(bboxes)
        bboxes = BoundingBoxes(bboxes, format=format, canvas_size=canvas_size)
        return bboxes

    def _get_position_realtive_to_identity(self, fragment):
        """
        Computes the position of the current fragment wrt the identity. That is,
        the position is the normalized distance of the fragment from the start
        of the identity.
        """
        instances = self.identifier.get_instances(
            self.identifier.find_identity_id(fragment[0])
        )
        first, last = instances[0], instances[-1]

        first_frame_id = self.instance_ds.img_ann(first)["frame_id"]
        last_frame_id = self.instance_ds.img_ann(last)["frame_id"]

        identity_length = last_frame_id - first_frame_id

        current_frame_id = self.instance_ds.img_ann(fragment[0])["frame_id"]

        pos = current_frame_id - first_frame_id
        pos_norm = pos / identity_length

        return torch.tensor(pos_norm)

    def _get_position_realtive_to_video(self, fragment):
        video_length, (first, last) = self._video_length[
            self.videoer.get_video_id(fragment[0])
        ]

        current_frame_id = self.instance_ds.img_ann(fragment[0])["frame_id"]

        pos = current_frame_id - first
        pos_norm = pos / video_length

        return torch.tensor(pos_norm)

    def _get_category_id(self, fragment):
        category_ids = [
            self.instance_ds.tgt_ann(frame_id).get("category_id")
            for frame_id in fragment
        ]

        assert (
            len(set(category_ids)) == 1
        ), "All category_ids in the fragment should be the same"

        category_id = list(set(category_ids))[0]

        if category_id is None:
            return torch.tensor(-1)  # unknown

        return torch.tensor(category_id)

    def _get_size(self, fragment):
        sizes = [
            self.instance_ds.tgt_ann(frame_id).get("size") for frame_id in fragment
        ]

        assert len(set(sizes)) == 1, "All sizes in the fragment should be the same"

        size = list(set(sizes))[0]

        return torch.tensor(size) if size is not None else torch.tensor(-1)  # unknown

    def _get_histology(self, fragment):
        histologies = [
            self.instance_ds.tgt_ann(frame_id).get("histology") for frame_id in fragment
        ]

        assert (
            len(set(histologies)) == 1
        ), "All histologies in the fragment should be the same"

        histology = list(set(histologies))[0]

        if histology is None:
            return torch.tensor(-1)  # unknown

        histology_label = self.histology2label[histology]

        return torch.tensor(histology_label)

    def _has_lesion(self, fragment):
        has_lesion = [
            self.instance_ds.img_ann(frame_id).get("has_lesion")
            for frame_id in fragment
        ]

        assert len(set(has_lesion)) == 1, "Fragment should either show or not a lesion"

        has_lesion = list(set(has_lesion))[0]

        if has_lesion is None:
            return torch.tensor(-1)  # unknown

        return torch.tensor(has_lesion)

    def _build(self):
        """
        Construct the list of fragments and labels, where labels have been
        mapped from identities to integers.
        """
        self.x = self.fragmenter.fragments()
        self.ids = list(range(len(self.x)))

        get_instance_id = lambda fragment_id: self.fragmenter.fragment(fragment_id)[0]

        identities = [
            self.identifier.find_identity_id(get_instance_id(fragment_id))
            for fragment_id in self.ids
        ]
        identities_unique = sorted(list(set(identities)))

        self.identity2label = {
            identity: idx for idx, identity in enumerate(identities_unique)
        }
        self.label2identity = {
            idx: identity for identity, idx in self.identity2label.items()
        }

        self.y = torch.tensor(
            [self.identity2label[identity] for identity in identities]
        )

        videos = [
            self.videoer.get_video_id(get_instance_id(fragment_id))
            for fragment_id in self.ids
        ]
        videos_unique = sorted(list(set(videos)))

        self._video_length = {
            video_id: self.videoer.get_video_length(video_id)
            for video_id in videos_unique
        }

        self.video2label = {video: idx for idx, video in enumerate(videos_unique)}
        self.label2video = {idx: video for video, idx in self.video2label.items()}
        self.v = torch.tensor([self.video2label[video] for video in videos])

        self.histology2label = {
            "OTHER": -1,
            "NO POLYP": -1,
            "HP": 0,
            "SSL": 1,
            "TSA": 1,
            "AD": 1,
        }
        self.label2histology = {
            idx: histology for histology, idx in self.histology2label.items()
        }

        self.pv = torch.tensor(
            [self._get_position_realtive_to_video(fragment) for fragment in self.x]
        )
        self.pi = torch.tensor(
            [self._get_position_realtive_to_identity(fragment) for fragment in self.x]
        )

        self.q = torch.tensor([self._has_lesion(fragment) for fragment in self.x])
