"""Test search on a public face photo."""
import cv2
import insightface.data
from dotenv import load_dotenv

load_dotenv()
from pipeline.ipfs_store import pin_file, gateway_url
from pipeline.search import reverse_image_search

def run_search_test():
    # Save public test face image
    img = insightface.data.get_image("Tom_Hanks_54745")
    cv2.imwrite("demo/public_test_face.jpg", img)

    cid = pin_file("demo/public_test_face.jpg", name="public_test_face.jpg")
    url = gateway_url(cid)
    print(f"IPFS CID: {cid}")
    print(f"Gateway URL: {url}")

    candidates = reverse_image_search(url)
    print(f"Discovered candidates: {len(candidates)}")
    for idx, c in enumerate(candidates[:10], start=1):
        print(f"#{idx} [{c['platform']}] {c['domain']} | {c['title'][:45]} | {c['link']}")


if __name__ == "__main__":
    run_search_test()
