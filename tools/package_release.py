"""Build a V6 source release without runtime data or bytecode."""
from pathlib import Path
import hashlib
import json
import zipfile


def main():
    root=Path(__file__).resolve().parents[1]
    destination=root.parent/'output';destination.mkdir(exist_ok=True)
    paths=sorted(p.relative_to(root) for p in root.rglob('*') if p.is_file()
                 and p.name!='RELEASE_SHA256.json'
                 and not any(part.startswith('.') or part in ('__pycache__','EvolutionVault','Analysis','output')
                             for part in p.relative_to(root).parts)
                 and p.suffix not in ('.pyc','.building','.zip'))
    manifest={p.as_posix():hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}
    manifest_json=json.dumps(manifest,indent=2,sort_keys=True)+'\n'
    with (root/'RELEASE_SHA256.json').open('w',encoding='utf-8',newline='\n') as stream:
        stream.write(manifest_json)
    archive=destination/'predatore_v6_05_eva_smith.zip'
    temporary=archive.with_suffix('.building')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in paths:z.write(root/p,'predatore/'+p.as_posix())
        z.writestr('predatore/RELEASE_SHA256.json',manifest_json)
    with zipfile.ZipFile(temporary) as z:
        assert z.testzip() is None
        assert {Path(n).parts[0] for n in z.namelist()} == {'predatore'}
        assert set(z.namelist()) == {'predatore/'+p.as_posix() for p in paths} | {'predatore/RELEASE_SHA256.json'}
        assert json.loads(z.read('predatore/RELEASE_SHA256.json')) == manifest
        for path,digest in manifest.items():
            assert hashlib.sha256(z.read('predatore/'+path)).hexdigest()==digest,path
    temporary.replace(archive)
    print(json.dumps({'file':str(archive),'bytes':archive.stat().st_size,'files':len(paths),
                      'sha256':hashlib.sha256(archive.read_bytes()).hexdigest()}))
if __name__=='__main__':main()
