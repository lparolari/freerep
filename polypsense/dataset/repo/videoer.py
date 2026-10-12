class Videoer:
    """
    A utility class that provide access to sequence-level metadata from single
    instances.

    The length of a video is determined by its actual instances instead of its
    absolute length. Specifically, we compute the video length by
        last_frame_id - first_frame_id

    The sequence identifier is extracted following these rules:

      1. Read from instance's image annotation, using the `video_key`

      2. Parsed automatically from instance's identity id:
        - If the identity id contains dataset0, dataset1, dataset2 then the
          identity itself is returned.
        - If the identity id contains dataset3, then the video id is extracted
          assuming REAL-Colon video id encoding. The dataset identifier is ketp.
          E.g. dataset3_004-005_1 -> dataset3_004-005
        - Else, then the video id is extracted assuming REAL-Colon video id
          encoding.
    """

    def __init__(self, ds, identity_key="identity_id", video_key="sequence_id"):
        self.ds = ds
        self.identity_key = identity_key
        self.video_key = video_key

        self.videoid2instanceid = {}
        self.instanceid2videoid = {}

        for i in range(len(self.ds)):
            id = self.ds.id(i)

            video_id = self._extract_video_id(id)

            if video_id not in self.videoid2instanceid:
                self.videoid2instanceid[video_id] = []

            self.videoid2instanceid[video_id].append(id)
            self.instanceid2videoid[id] = video_id

    def get_video_id(self, instance_id):
        """
        Return the video identifier from the instance id.
        """
        return self.instanceid2videoid[instance_id]

    def get_video_length(self, video_id):
        """
        Return the video length defined on the actual instances, not the
        absolute length of video:
            last_frame_id - first_frame_id
        """
        instance_ids = self.get_instance_ids_by_video_id(video_id)

        first, last = instance_ids[0], instance_ids[-1]

        first_frame_id = self.ds.img_ann(first)["frame_id"]
        last_frame_id = self.ds.img_ann(last)["frame_id"]

        return last_frame_id - first_frame_id, (first_frame_id, last_frame_id)

    def get_instance_ids_by_video_id(self, video_id):
        """
        Return the list of instances associated to given video id sorted by
        frame id.
        """
        inst_ids = self.videoid2instanceid[video_id]
        inst_img_anns = [self.ds.img_ann(inst_id) for inst_id in inst_ids]

        frame_ids = [inst["frame_id"] for inst in inst_img_anns]

        idx = range(len(inst_ids))
        sorted_idx = sorted(idx, key=lambda i: frame_ids[i])

        return [inst_ids[i_sorted] for i_sorted in sorted_idx]

    def list_video_ids(self):
        """
        Return all video ids in the dataset.
        """
        return list(self.videoid2instanceid.keys())

    def _extract_video_id(self, instance_id):
        img_ann = self.ds.img_ann(instance_id)

        video_id = img_ann.get(self.video_key)

        if video_id:
            return video_id

        # fallback to parsing from identity_id
        tgt_ann = self.ds.tgt_ann(instance_id)
        identity_id = tgt_ann.get(self.identity_key)

        video_id = parse_video_id_from_identity_id(identity_id)

        if video_id:
            return video_id

        raise ValueError(f"Could not parse video_id from identity_id ({identity_id})")


def parse_video_id_from_identity_id(identity_id):
    if not identity_id:
        return None

    if (
        "dataset0" in identity_id
        or "dataset1" in identity_id
        or "dataset2" in identity_id
    ):
        video_id = identity_id

    elif "dataset3" in identity_id:
        # e.g. dataset3_004-005_1, drop _1 and take the others
        video_id = "_".join(identity_id.split("_")[:-1])

    else:
        # legacy, i.e. realcolon only (v_id)
        video_id = identity_id.split("_")[0]

    return video_id
