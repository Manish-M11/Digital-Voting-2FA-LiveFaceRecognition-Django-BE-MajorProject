from django.shortcuts import render
from django.contrib import messages
from EC_Admin.models import Voters, Candidates, Election, Votes, Reports
from django.contrib.auth.models import User, auth
from django.http import JsonResponse
import requests
import datetime
from .models import Voted, Complain
from django.contrib.auth.decorators import login_required
from Digital_Voting.settings import BASE_DIR
from django.core.mail import send_mail
import math
import random

import cv2
import os
import numpy as np
from PIL import Image
from django.conf import settings
from django.core.files.storage import FileSystemStorage
import itertools

# Create your views here.
import face_recognition
import cv2
import os
import glob
import numpy as np


class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

        # Resize frame for a faster speed
        self.frame_resizing = 0.25

    def load_encoding_images(self, images_path):
        """
        Load encoding images from path
        :param images_path:
        :return:
        """
        # Load Images
        images_path = glob.glob(os.path.join(images_path, "*.*"))

        print("{} encoding images found.".format(len(images_path)))

        # Store image encoding and names
        for img_path in images_path:
            img = cv2.imread(img_path)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Get the filename only from the initial file path.
            basename = os.path.basename(img_path)
            (filename, ext) = os.path.splitext(basename)
            # Get encoding
            img_encoding = face_recognition.face_encodings(rgb_img)[0]

            # Store file name and file encoding
            self.known_face_encodings.append(img_encoding)
            self.known_face_names.append(filename)
        print("Encoding images loaded")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(
            frame, (0, 0), fx=self.frame_resizing, fy=self.frame_resizing)
        # Find all the faces and face encodings in the current frame of video
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(
            rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(
                self.known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(
                self.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = self.known_face_names[best_match_index]
            face_names.append(name)

        # Convert to numpy array to adjust coordinates with frame resizing quickly
        face_locations = np.array(face_locations)
        face_locations = face_locations / self.frame_resizing
        return face_locations.astype(int), face_names


def register_vid(request):
    if (request.method == 'POST'):
        voterid = request.POST['vid']
        if (User.objects.filter(username=voterid)).exists():
            messages.info(request, 'Voter already registered')
            return render(request, 'registervid.html')
        else:
            if Voters.objects.filter(voterid_no=voterid):
                register_vid.v = Voters.objects.get(voterid_no=voterid)
                user_phone = str(register_vid.v.mobile_no)
                url = "http://2factor.in/API/V1/a8e15c6b-a550-11ed-813b-0200cd936042/SMS/" + \
                    user_phone + "/AUTOGEN"
                response = requests.request("GET", url)
                data = response.json()
                request.session['otp_session_data'] = data['Details']
                messages.info(
                    request, 'an OTP has been sent to registered mobile number ending with')
                mobno = user_phone[6:]
                return render(request, 'otp.html', {'mno': mobno})
            else:
                messages.info(request, 'Invalid Voter ID')
                return render(request, 'registervid.html')


def otp(request):
    if (request.method == "POST"):
        userotp = request.POST['otp']
        url = "http://2factor.in/API/V1/a8e15c6b-a550-11ed-813b-0200cd936042/SMS/VERIFY/" + request.session[
            'otp_session_data'] + "/" + userotp
        response = requests.request("GET", url)
        data = response.json()
        if data['Status'] == "Success":
            response_data = {'Message': 'Success'}
            return render(request, './register.html',
                          {'voterid_no': register_vid.v.voterid_no, 'name': register_vid.v.name,
                           'father_name': register_vid.v.father_name, 'gender': register_vid.v.gender,
                           'dateofbirth': register_vid.v.dateofbirth, 'address': register_vid.v.address,
                           'mobile_no': register_vid.v.mobile_no, 'state': register_vid.v.state,
                           'pincode': register_vid.v.pincode, 'parliamentary': register_vid.v.parliamentary,
                           'assembly': register_vid.v.assembly, 'voter_image': register_vid.v.voter_image})
        else:
            messages.info(request, 'Invalid OTP')
            return render(request, 'otp.html')


def register(request):
    if (request.method == 'POST'):
        voter_id = request.POST.get('v_id')
        email = request.POST['email']
        password1 = request.POST['password1']
        password2 = request.POST['password2']
        vidfile = request.FILES['vidfile']
        v = Voters.objects.get(voterid_no=voter_id)
        Id = str(v.id)
        folder = BASE_DIR+"/DatasetVideo/"
        fs = FileSystemStorage(location=folder)
        vidfilename = Id+vidfile.name
        filename = fs.save(vidfilename, vidfile)
        name = v.voterid_no
        faceDetect = cv2.CascadeClassifier(
            BASE_DIR + "/haarcascade_frontalface_default.xml")
        cam = cv2.VideoCapture(folder+"/"+vidfilename)
        # print(cam)
        sampleNum = 0
        while (True):
            ret, img = cam.read()
            if not ret:
                print("no image")
                break
            # print(img, ret)
            if (img is not None):
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # print(gray, ret)
            faces = faceDetect.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                sampleNum = sampleNum + 1
                cv2.imwrite(BASE_DIR + "/TrainingImage/" + name + "." +
                            Id + '.' + str(sampleNum) + ".jpg", gray[y:y + h, x:x + w])
                cv2.imshow("Face", img)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            elif sampleNum > 60:
                break
        cam.release()
        cv2.destroyAllWindows()

        # Train
        recognizer = cv2.face.LBPHFaceRecognizer_create()

        def getImagesAndLabels(path):
            imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
            faces = []
            Ids = []
            for imagePath in imagePaths:
                pilImage = Image.open(imagePath).convert('L')
                imageNp = np.array(pilImage, 'uint8')
                Id = int(os.path.split(imagePath)[-1].split('.')[1])
                faces.append(imageNp)
                Ids.append(Id)
            return faces, Ids
        faces, Id = getImagesAndLabels(BASE_DIR + "/TrainingImage/")
        recognizer.train(faces, np.array(Id))
        recognizer.save(BASE_DIR + "/TrainingImageLabel/Trainner.yml")
        if password1 == password2:
            add_user = User.objects.create_user(
                username=voter_id, password=password1, email=email)
            add_user.save()
            messages.info(request, 'Voter Registered')
            return render(request, 'index.html')


# def register2(request):
#     # Load the cascade classifier
#     face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#     # Load the target image for face recognition
#     target_image = cv2.imread('target_face.jpg', 0)
#     target_face = face_cascade.detectMultiScale(target_image, 1.3, 5)

#     # If target face is not found, print an error message and exit the program
#     if len(target_face) == 0:
#         print("Target face not found")
#         exit()

#     # Extract the target face and compute its encoding
#     target_face = target_face[0]
#     target_encoding = cv2.face.MTCNN().compute_face_descriptor(
#         target_image, target_face)[0]

#     # Start capturing the video
#     cap = cv2.VideoCapture(0)
#     while True:
#         # Read the video frame
#         ret, frame = cap.read()
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         # Detect faces in the frame
#         faces = face_cascade.detectMultiScale(gray, 1.3, 5)

#         # Iterate over the detected faces
#         for (x, y, w, h) in faces:
#             # Extract the face and compute its encoding
#             face = frame[y:y+h, x:x+w]
#             face_encoding = cv2.face.MTCNN().compute_face_descriptor(
#                 frame, [y, x, h, w])[0]

#             # Compute the Euclidean distance between the encodings
#             distance = np.linalg.norm(target_encoding - face_encoding)

#             # If the distance is less than a threshold, draw a green rectangle around the face
#             if distance < 0.5:
#                 cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

#         # Show the frame
#         cv2.imshow("Face recognition", frame)

#         # Break the loop if the 'q' key is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # Release the video capture and destroy all windows
#     cap.release()
#     cv2.destroyAllWindows()


@login_required(login_url='home')
def vhome(request):
    vhome.username = request.session['v_id']
    v = Voters.objects.get(voterid_no=vhome.username)
    vhome.image = v.voter_image
    return render(request, 'voter/vhome.html', {'username': vhome.username, 'image': vhome.image})


@login_required(login_url='home')
def vprocess(request):
    return render(request, 'voter/votingprocess.html', {'username': vhome.username, 'image': vhome.image})


@login_required(login_url='home')
def vprofile(request):
    v_id = request.session['v_id']
    v = Voters.objects.get(voterid_no=v_id)
    vemail = User.objects.get(username=v_id)
    return render(request, 'voter/voter profile.html', {'voterid_no': v.voterid_no, 'name': v.name,
                                                        'father_name': v.father_name, 'gender': v.gender,
                                                        'dateofbirth': v.dateofbirth, 'address': v.address,
                                                        'mobile_no': v.mobile_no, 'state': v.state,
                                                        'pincode': v.pincode, 'parliamentary': v.parliamentary,
                                                        'assembly': v.assembly, 'voter_image': v.voter_image,
                                                        'email': vemail.email, 'username': vhome.username, 'image': vhome.image})


@login_required(login_url='home')
def vchangepassword(request):
    return render(request, 'voter/vchangepassword.html', {'username': vhome.username, 'image': vhome.image})


@login_required(login_url='home')
def vchange_password(request):
    if request.method == "POST":
        v_id = request.session['v_id']
        oldpass = request.POST['oldpass']
        newpass = request.POST['password1']
        newpass2 = request.POST['password2']
        u = auth.authenticate(username=v_id, password=oldpass)
        if u is not None:
            u = User.objects.get(username=v_id)
            if oldpass != newpass:
                if newpass == newpass2:
                    u.set_password(newpass)
                    u.save()
                    messages.info(request, 'Password Changed')
                    return render(request, 'voter/vchangepassword.html', {'username': vhome.username, 'image': vhome.image})
            else:
                messages.info(request, 'New password is same as old password')
                return render(request, 'voter/vchangepassword.html', {'username': vhome.username, 'image': vhome.image})
        else:
            messages.info(request, 'Old Password not matching')
            return render(request, 'voter/vchangepassword.html', {'username': vhome.username, 'image': vhome.image})


@login_required(login_url='home')
def vviewcandidate(request):
    return render(request, 'voter/view candidate.html', {'username': vhome.username, 'image': vhome.image})


@login_required(login_url='home')
def vview_candidate(request):
    if request.method == 'POST':
        state = request.POST['states']
        constituency1 = request.POST['constituency1']
        constituency2 = request.POST['constituency2']
        if constituency1 == 'Parliamentary':
            candidates = Candidates.objects.filter(
                state=state, constituency=constituency1, parliamentary=constituency2)
            if candidates:
                return render(request, 'voter/view candidate.html', {'constituency': constituency2, 'candidates': candidates, 'username': vhome.username, 'image': vhome.image})
            else:
                messages.info(request, 'No Candidate Found')
                return render(request, 'voter/view candidate.html', {'username': vhome.username, 'image': vhome.image})
        else:
            candidates = Candidates.objects.filter(
                state=state, constituency=constituency1, assembly=constituency2)
            if candidates:
                return render(request, 'voter/view candidate.html', {'constituency': constituency2, 'candidates': candidates, 'username': vhome.username, 'image': vhome.image})
            else:
                messages.info(request, 'No Candidate Found')
                return render(request, 'voter/view candidate.html', {'username': vhome.username, 'image': vhome.image})


@login_required(login_url='home')
def velection(request):
    v_id = request.session['v_id']
    vdetail = Voters.objects.get(voterid_no=v_id)
    status = 'active'
    if Election.objects.filter(state=vdetail.state, status=status):
        velection.e = Election.objects.get(state=vdetail.state, status=status)
        now = datetime.datetime.now()
        nowdate = now.strftime("%G-%m-%d")
        nowtime = now.strftime("%X")
        sdate = str(velection.e.start_date)
        if nowdate == sdate:
            stime = str(velection.e.start_time)
            etime = str(velection.e.end_time)
            if stime <= nowtime <= etime:
                if velection.e.election_type == 'PC-GENERAL':
                    vpc = vdetail.parliamentary
                    candidates = Candidates.objects.filter(parliamentary=vpc)
                    return render(request, 'voter/velection.html', {'candidate': candidates, 'username': vhome.username, 'image': vhome.image})
                elif velection.e.election_type == 'AC-GENERAL':
                    vac = vdetail.assembly
                    candidates = Candidates.objects.filter(assembly=vac)
                    return render(request, 'voter/velection.html', {'candidate': candidates, 'username': vhome.username, 'image': vhome.image})
            else:
                messages.info(request, 'No Elections Runnning')
                return render(request, 'voter/vnoelection.html', {'username': vhome.username, 'image': vhome.image})
        else:
            messages.info(request, 'No Elections Runnning')
            return render(request, 'voter/vnoelection.html', {'username': vhome.username, 'image': vhome.image})
    else:
        messages.info(request, 'No Elections Runnning')
        return render(request, 'voter/vnoelection.html', {'username': vhome.username, 'image': vhome.image})


@login_required(login_url='home')
def vote(request):
    if request.method == "POST":
        v_id = request.session['v_id']
        vote.v = Voted.objects.get(
            election_id=velection.e.election_id, voter_id=v_id)

        print(vote.v)

        if vote.v.has_voted == 'no':
            vidofv = Voters.objects.get(voterid_no=v_id)
            detectuserid = str(vidofv.id)  # vidofv = voter id
            # vidfile = request.FILES['vidfile']  # to change
            sfr = SimpleFacerec()
            sfr.load_encoding_images("media\VoterImage")

            folder = BASE_DIR+"/VotingDSVideo/"  # to change
            fs = FileSystemStorage(location=folder)
            # vidfilename = detectuserid+vidfile.name
            # filename = fs.save(vidfilename, vidfile)
            # rec = cv2.face.LBPHFaceRecognizer_create()  # //
            # rec.read(BASE_DIR+"/TrainingImageLabel/Trainner.yml")  # //
            # faceDetect = cv2.CascadeClassifier(
            #     BASE_DIR+"/haarcascade_frontalface_default.xml")
            # cam = cv2.VideoCapture(folder+"/"+vidfilename)  # //
            #cap = cv2.VideoCapture(0)
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            # font = cv2.FONT_HERSHEY_SIMPLEX
            flag = 0
            unknowncount = 0
            # while flag != 1 and unknowncount != 5:
            #     ret, img = cam.read()
            #     if not ret:
            #         print("no image")
            #         break
            #     # print(img, ret)
            #     if (img is not None):
            #         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #     # print(gray, ret)
            #     faces = faceDetect.detectMultiScale(gray, 1.3, 5)
            #     for(x, y, w, h) in faces:
            #         cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            #         Id, conf = rec.predict(gray[y:y+h, x:x+w])
            #         if conf < 50:  # if matched then this
            #             if str(Id) == detectuserid:
            #                 tt = "Detected"
            #                 cv2.putText(img, str(tt), (x, y+h),
            #                             font, 2, (0, 255, 0), 2)
            #                 cv2.waitKey(500)
            #                 flag = 1
            #         else:
            #             Id = 'Unknown'
            #             unknowncount += 1
            #             tt = str(Id)
            #             cv2.putText(img, str(tt), (x, y+h),
            #                         font, 2, (0, 0, 255), 2)
            #     cv2.imshow("Face", img)
            #     cv2.waitKey(1000)
            #     if(cv2.waitKey(1) == ord('q')):
            #         break
            # cam.release()
            # cv2.destroyAllWindows()
            while True:
                ret, frame = cap.read()

                # Detect Faces
                face_locations, face_names = sfr.detect_known_faces(frame)
                # print(len(face_names))
                for face_loc, name in zip(face_locations, face_names):
                    y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

                    cv2.putText(frame, name, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

                    if face_names[0] == str(v_id):
                        flag = 1
                        break
                    # else:
                    #     Id = 'Unknown'
                    #     unknowncount += 1
                    #     tt = str(Id)

                cv2.imshow("Frame", frame)

                if(cv2.waitKey(1) == ord('q')):
                    break

            cap.release()
            cv2.destroyAllWindows()
            print(flag)
            # flag = 1
            if flag == 1:

                vote.candidateid = request.POST['can']
                vmob = Voters.objects.get(voterid_no=v_id)
                vmobno = str(vmob.mobile_no)
                url = "http://2factor.in/API/V1/a8e15c6b-a550-11ed-813b-0200cd936042/SMS/" + \
                    vmobno + "/AUTOGEN"
                response = requests.request("GET", url)
                data = response.json()
                request.session['otp_session_data'] = data['Details']
                response_data = {'Message': 'Success'}
                messages.info(
                    request, 'face matched and otp send')
                mobno = vmobno[6:]
                return render(request, 'voter/voteotp.html', {'mno': mobno, 'username': vhome.username, 'image': vhome.image})
                # messages.info(request, 'Face matched')
                # return render(request, 'voter/voteotp.html', {'username': vhome.username, 'image': vhome.image})
            else:
                messages.info(request, 'Face... not matched')
                return render(request, 'voter/votesub.html', {'username': vhome.username, 'image': vhome.image})
        else:
            messages.info(request, 'Already Voted')
            return render(request, 'voter/votesub.html', {'username': vhome.username, 'image': vhome.image})


@login_required(login_url='home')
def subvoteotp(request):
    if (request.method == "POST"):
        userotp = request.POST['otp']
        url = "http://2factor.in/API/V1/a8e15c6b-a550-11ed-813b-0200cd936042/SMS/VERIFY/" + \
            request.session['otp_session_data'] + "/" + userotp
        response = requests.request("GET", url)
        data = response.json()
        if data['Status'] == "Success":
            v_id = request.session['v_id']
            vemail = User.objects.get(username=v_id)
            email = vemail.email
            string = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
            length = len(string)
            subvoteotp.otp = ""
            for i in range(6):
                subvoteotp.otp += string[math.floor(random.random()*length)]
            send_mail(
                'OTP from Digital Voting',
                'The following is the 6 digit alphanumeric email OTP to be entered for vote submission '+subvoteotp.otp,
                settings.EMAIL_HOST_USER,
                [email]
            )
            emailstart = email[0:3]
            emailend = email[-13:]
            emailid = emailstart+'*****'+emailend
            messages.info(
                request, 'an 6 digit alphanumeric OTP has been sent to your registered email address ')
            return render(request, 'voter/voteemailotp.html', {'username': vhome.username, 'image': vhome.image, 'email': emailid})
        else:
            messages.info(request, 'Invalid OTP')
            return render(request, 'voter/voteotp.html', {'username': vhome.username, 'image': vhome.image})


@login_required(login_url='home')
def subvoteemailotp(request):
    if request.method == "POST":
        emailotp = request.POST['emailotp']
        if subvoteotp.otp == emailotp:
            votecan = Votes.objects.get(
                election_id=velection.e.election_id, candidate_id=vote.candidateid)
            votecan.online_votes += 1
            votecan.save()
            vote.v.has_voted = 'yes'
            vote.v.where_voted = 'online'
            x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
            if x_forwarded_for:
                ipaddress = x_forwarded_for.split(',')[-1].strip()
            else:
                ipaddress = request.META.get('REMOTE_ADDR')
            vote.v.ipaddress = ipaddress
            vote.v.datetime = datetime.datetime.now()
            vote.v.save()
            messages.info(request, 'Vote submitted to ')
            return render(request, 'voter/votesub.html', {'votesub': votecan.candidate_name, 'username': vhome.username, 'image': vhome.image})
        else:
            messages.info(request, 'Invalid OTP')
            return render(request, 'voter/voteemailotp.html', {'username': vhome.username, 'image': vhome.image})


@login_required(login_url='home')
def vviewresult(request):
    elections = Election.objects.all()
    return render(request, 'voter/viewresult.html', {'elections': elections, 'username': vhome.username, 'image': vhome.image})


@login_required(login_url='home')
def vview_result(request):
    if request.method == "POST":
        election_id = request.POST['e_id']
        resulttype = request.POST['resulttype']
        e = Election.objects.get(election_id=election_id)
        estate = e.state
        if resulttype == "partywise":
            result = Votes.objects.filter(election_id=election_id)
            v = Votes.objects.filter(election_id=election_id)
            constituencies = []
            for i in v:
                if i.constituency not in constituencies:
                    constituencies.append(i.constituency)
            d = []
            for i in constituencies:
                resultcon = Votes.objects.filter(
                    election_id=election_id, constituency=i)
                maxi = 0
                for i in resultcon:
                    if i.total_votes > maxi:
                        maxi = i.total_votes
                        d.append(i.candidate_party)
            parties = []
            for k in v:
                if k.candidate_party not in parties:
                    parties.append(k.candidate_party)
            final = {}
            for i in parties:
                c = d.count(i)
                final.update({i: c})
            par = []
            won = []
            for k, v in final.items():
                par.append(k)
                won.append(v)
            parwon = zip(par, won)
            total = 0
            for i in won:
                total += i
            return render(request, 'voter/viewpartywise.html', {'total': total, 'parwon': parwon, 'electionid': election_id, 'state': estate, 'username': vhome.username, 'image': vhome.image})
        elif resulttype == "constituencywise":
            v = Votes.objects.filter(election_id=election_id)
            constituencies = []
            for i in v:
                if i.constituency not in constituencies:
                    constituencies.append(i.constituency)
            return render(request, 'voter/viewresultconwise.html', {'electionid': election_id, 'state': estate, 'constituency': constituencies, 'username': vhome.username, 'image': vhome.image})


@login_required(login_url='home')
def vview_result_filter(request):
    if request.method == "POST":
        election_id = request.POST['e_id']
        constituency = request.POST['constituency']
        e = Election.objects.get(election_id=election_id)
        estate = e.state
        v = Votes.objects.filter(election_id=election_id)
        result = Votes.objects.filter(
            election_id=election_id, constituency=constituency)
        constituencies = []
        for i in v:
            if i.constituency not in constituencies:
                constituencies.append(i.constituency)
        totalvotes = 0
        totalonline = 0
        totalevm = 0
        for i in result:
            totalvotes += i.total_votes
            totalonline += i.online_votes
            totalevm += i.evm_votes
        perofvotes = []
        for i in result:
            per = (i.total_votes/totalvotes)*100
            percentage = float("{:.2f}".format(per))
            perofvotes.append(percentage)
        finalresult = zip(result, perofvotes)
        return render(request, 'voter/viewresultconwise.html', {'totalonline': totalonline, 'totalevm': totalevm, 'totalvotes': totalvotes, 'result': finalresult, 'electionid': election_id, 'state': estate, 'constituency': constituencies, 'username': vhome.username, 'image': vhome.image})


@login_required(login_url='home')
def vviewreport(request):
    elections = Election.objects.all()
    return render(request, 'voter/viewreport.html', {'elections': elections, 'username': vhome.username, 'image': vhome.image})


@login_required(login_url='home')
def vview_report(request):
    election_id = request.POST['e_id']
    constituency = request.POST['constituency2']
    report = Reports.objects.filter(
        election_id=election_id, constituency=constituency)
    elections = Election.objects.all()
    return render(request, 'voter/viewreport.html', {'report': report, 'elections': elections, 'username': vhome.username, 'image': vhome.image})


@login_required(login_url='home')
def vcomplain(request):
    v_id = request.session['v_id']
    complain = Complain.objects.filter(
        voterid_no=v_id, viewed=True, replied=True)
    return render(request, 'voter/vcomplain.html', {'voterid_no': v_id, 'reply': complain, 'username': vhome.username, 'image': vhome.image})


@login_required(login_url='home')
def submitcomplain(request):
    if (request.method == 'POST'):
        v_id = request.session['v_id']
        complain = request.POST['complain']
        addcomplain = Complain(voterid_no=v_id, complain=complain)
        addcomplain.save()
        messages.info(request, 'complain submitted')
        return render(request, 'voter/vcomplain.html', {'username': vhome.username, 'image': vhome.image})
