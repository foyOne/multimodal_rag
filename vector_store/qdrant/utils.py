from embeddings import Distance as SourceDistance
from qdrant_client.models import Distance as TargetDistance


def convert_to_qdrant_distance(
    base_distance: SourceDistance,
) -> TargetDistance:
    match base_distance:
        case SourceDistance.COSINE:
            return TargetDistance.COSINE
        case SourceDistance.EUCLID:
            return TargetDistance.EUCLID
        case SourceDistance.DOT:
            return TargetDistance.DOT
        case SourceDistance.MANHATTAN:
            return TargetDistance.MANHATTAN
        case _:
            raise Exception("There is no other target distance options")
