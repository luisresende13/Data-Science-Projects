import sys, requests, os, json

# Job-defined env vars
TASK_INDEX = os.getenv("CLOUD_RUN_TASK_INDEX", 0)
TASK_ATTEMPT = os.getenv("CLOUD_RUN_TASK_ATTEMPT", 0)

api_root = 'https://bolsao-api-j2fefywbia-rj.a.run.app'

urls = [
    '/mongo/post/predict/live?db=Waterbag&coll=Prediction&overwrite=False&as_datetime=timestamp',
    '/mongo/post/city?db=Waterbag&coll=City&overwrite=True',
    '/mongo/post/clusters?db=Waterbag&coll=Polygons&overwrite=False',
    '/mongo/post/clusters/overview?db=Waterbag&coll=Polygons Overview&overwrite=True',
    '/mongo/post/ipp/polygons?db=Waterbag&coll=Polygons IPP&overwrite=True',
    '/mongo/post/ipp/polygons/overview?db=Waterbag&coll=Polygons IPP Overview&overwrite=True',
    '/mongo/post/polygons/alertario?db=Waterbag&coll=Polygons AlertaRio&overwrite=True',
    '/mongo/post/cameras/alertario/stations?db=Waterbag&coll=Cameras AlertaRio&overwrite=True',    
]

def main():
    print(f"Starting Task #{TASK_INDEX}, Attempt #{TASK_ATTEMPT}...")
    inline_requests(urls)
    print(f"Completed Task #{TASK_INDEX}.")

def inline_requests(urls):
    fail = {}
    for url in urls:
        res = requests.get(api_root + url)
        if not res:
            fail[url] = res.status_code
            print(f'STATUS CODE {fail[url]} - {url}')
    if len(fail) == len(urls): print(f'MONGO POSTS ALL FAILED. NOT POSTED: {len(fail)}')
    elif len(fail): print(f'MONGO POSTS INCOMPLETE. NOT POSTED: {len(fail)}')
    else: print(f'MONGO POSTS SUCCESS. POSTED {len(urls)}')
    return

if __name__ == "__main__":
    try:
        main()
    except Exception as err:
        message = f"Task #{TASK_INDEX}, " \
                  + f"Attempt #{TASK_ATTEMPT} failed: {str(err)}"
        print(json.dumps({"message": message, "severity": "ERROR"}))
        sys.exit(1)  # Retry Job Task by exiting the process