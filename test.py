from session import Session
from util import SessionManager

ses = Session()
ses.add_report("amd", '2022', '10k', file_location='.')
ses_dict = ses.to_dict()
ses2 = Session.from_dict(ses_dict)
ses2_dict = ses2.to_dict()
print(ses_dict)
print("="*20)
print(ses2_dict)
sm = SessionManager()
sm.load()
for s in sm.sessions.values():
    print(s.to_dict())
# sm.add_session(ses)

# sm.save()

