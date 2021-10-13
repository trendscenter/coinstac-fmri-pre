import ujson as json,sys,os

sys.stdout.write(json.dumps({
        "output": {
            "message": "dummry_remote"
        },
        "cache": {},
        "success": True
    }))
