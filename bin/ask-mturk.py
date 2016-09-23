#!/usr/bin/env python

from boto.mturk.connection import MTurkConnection
import boto.mturk.question as q
import math

ACCESS_ID = 'AKIAJXDPX2JYBTFPTHDQ'
SECRET_KEY = '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'
HOST = 'mechanicalturk.sandbox.amazonaws.com'

 
mtc = MTurkConnection(aws_access_key_id=ACCESS_ID,
                      aws_secret_access_key=SECRET_KEY,
                      host=HOST)
 
print mtc.get_account_balance();

def make_simple_hit():
    title = 'Give your opinion about a website'
    description = ('Visit a website and give us your opinion about'
                   ' the design and also some personal comments')
    keywords = 'website, rating, opinions'
 
    ratings =[('Very Bad','-2'),
              ('Bad','-1'),
              ('Not bad','0'),
              ('Good','1'),
              ('Very Good','1')]

    overview = q.Overview()
    overview.append_field('Title', 'Give your opinion on this website')
    overview.append(q.FormattedContent('<a target="_blank"'
                                       ' href="http://www.toforge.com">'
                                       ' Mauro Rocco Personal Forge</a>'))
 
    qc1 = q.QuestionContent()
    qc1.append_field('Title','How looks the design ?')
    fta1 = q.SelectionAnswer(min=1, max=1,style='dropdown',
                             selections=ratings,
                             type='text',
                             other=False)
    q1 = q.Question(identifier='design',
                    content=qc1,
                    answer_spec=q.AnswerSpecification(fta1),
                    is_required=True)
 
 
    qc2 = q.QuestionContent()
    qc2.append_field('Title','Your personal comments')
    q2 = q.Question(identifier="comments",
                    content=qc2,
                    answer_spec=q.AnswerSpecification(q.FreeTextAnswer()))
 
    question_form = q.QuestionForm()
    question_form.append(overview)
    question_form.append(q1)
    question_form.append(q2)
 
    mtc.create_hit(questions=question_form,
                   max_assignments=1,
                   title=title,
                   description=description,
                   keywords=keywords,
                   duration = 60*5,
                   reward=0.05)

def make_external_hit(url, height):
    title = 'Use Kaizen'
    description = ('Look as several examples, and then choose similar images')
    keywords = 'image, features, similarity'
 
    eq = q.ExternalQuestion(url, height)
 
    return mtc.create_hit(question=eq,
                          max_assignments=2,
                          title=title,
                          description=description,
                          keywords=keywords,
                          duration = 60*15,
                          reward=0.06)

make_external_hit("https://www.jannotti.com/kaizen/dataset/", 1000)

def get_all_reviewable_hits(mtc):
    page_size = 100
    pn = 1
    hits = mtc.get_reviewable_hits(page_size=page_size, page_number=pn)
    print "Total results to fetch %s " % hits.TotalNumResults
    print "Request hits page %i" % 1
    total_pages = math.ceil(float(hits.TotalNumResults) / page_size)

    while pn < total_pages:
        pn = pn + 1
        print "Request hits page %i" % pn
        page_hits = mtc.get_reviewable_hits(page_size=page_size, page_number=pn)
        hits.extend(page_hits)
    return hits
 
reviewable = get_all_reviewable_hits(mtc)
 
for hit in reviewable:
    assignments = mtc.get_assignments(hit.HITId)
    for assignment in assignments:
        print "Answers of the worker %s" % assignment.WorkerId
        for answer in assignment.answers:
            for qfa in answer:
                print  "%s: %s" % (qfa.qid, qfa.fields)
#                for key, value in question_form_answer.fields:
#                    print "%s: %s" % (key,value)
        print "--------------------"

