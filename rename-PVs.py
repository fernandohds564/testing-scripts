#!/usr/bin/python3

import requests

URL = "https://10.0.38.42/mgmt/bpl"

def login(username, password):

    headers = { "User-Agent" : "Mozilla/5.0" }
    payload = { "username" : username, "password" : password }

    session = requests.Session()

    response = session.post(URL + "/login", headers = headers, data = payload, verify = False)

    if (b"authenticated" in response.content):
        return(session)
    else:
        return(None)

def deletePVs(session, PV_names):
    for PV_name in PV_names:
        session.get(URL + "/deletePV?pv=" + PV_name + "&deleteData=true") 

def pausePVs(session, PV_names):
    for PV_name in PV_names:
        session.get(URL + "/pauseArchivingPV?pv=" + PV_name)

def renamePV(session, old_PV_name, new_PV_name):
    session.get(URL + "/renamePV?pv=" + old_PV_name + "&newname=" + new_PV_name)

def resumePVs(session, PV_names):
    for PV_name in PV_names:
        session.get(URL + "/resumeArchivingPV?pv=" + PV_name)    

if (__name__ == "__main__"):

    session = login("eduardo.coelho", "123")

    pausePVs(session, ["sirius_md_02:READI", "sirius_md_02:READV"])

    renamePV(session, "sirius_md_02:READI", "LI-01:PU-Modltr-2:READI")
    renamePV(session, "sirius_md_02:READV", "LI-01:PU-Modltr-2:READV")

    resumePVs(session, ["LI-01:PU-Modltr-2:READI", "LI-01:PU-Modltr-2:READV"])

    deletePVs(session, ["sirius_md_02:READI", "sirius_md_02:READV"])
