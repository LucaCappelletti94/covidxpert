covidxpert
=========================================================================================
|travis| |sonar_quality| |sonar_maintainability| |codacy|
|code_climate_maintainability| |pip| |downloads|

Model to distinguish covid cases from viral pneumonia cases.

How do I install this package?
----------------------------------------------
As usual, just download it using pip:

.. code:: shell

    sudo apt update
    sudo apt install -y libsm6 libxext6
    sudo apt-get install -y gcc g++ libxrender-dev
    pip install covidxpert

Tests Coverage
----------------------------------------------
Since some software handling coverages sometimes
get slightly different results, here's three of them:

|coveralls| |sonar_coverage| |code_climate_coverage|

Trobleshooting errors
-----------------------------------------------

libGL errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you get the following error:

.. code:: shell

    libGL.so.1: cannot open shared object file: No such file or directory
    
You can solve it by running:

.. code:: shell
    
    apt-get update && apt-get install ffmpeg libsm6 libxext6  -y


.. |travis| image:: https://travis-ci.org/LucaCappelletti94/covidxpert.png
   :target: https://travis-ci.org/LucaCappelletti94/covidxpert
   :alt: Travis CI build

.. |sonar_quality| image:: https://sonarcloud.io/api/project_badges/measure?project=LucaCappelletti94_covidxpert&metric=alert_status
    :target: https://sonarcloud.io/dashboard/index/LucaCappelletti94_covidxpert
    :alt: SonarCloud Quality

.. |sonar_maintainability| image:: https://sonarcloud.io/api/project_badges/measure?project=LucaCappelletti94_covidxpert&metric=sqale_rating
    :target: https://sonarcloud.io/dashboard/index/LucaCappelletti94_covidxpert
    :alt: SonarCloud Maintainability

.. |sonar_coverage| image:: https://sonarcloud.io/api/project_badges/measure?project=LucaCappelletti94_covidxpert&metric=coverage
    :target: https://sonarcloud.io/dashboard/index/LucaCappelletti94_covidxpert
    :alt: SonarCloud Coverage

.. |coveralls| image:: https://coveralls.io/repos/github/LucaCappelletti94/covidxpert/badge.svg?branch=master
    :target: https://coveralls.io/github/LucaCappelletti94/covidxpert?branch=master
    :alt: Coveralls Coverage

.. |pip| image:: https://badge.fury.io/py/covidxpert.svg
    :target: https://badge.fury.io/py/covidxpert
    :alt: Pypi project

.. |downloads| image:: https://pepy.tech/badge/covidxpert
    :target: https://pepy.tech/project/covidxpert
    :alt: Pypi total project downloads

.. |codacy| image:: https://api.codacy.com/project/badge/Grade/a06342632e1a4e4b98f9a21edee318c3
    :target: https://www.codacy.com/manual/LucaCappelletti94/covidxpert?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=LucaCappelletti94/covidxpert&amp;utm_campaign=Badge_Grade
    :alt: Codacy Maintainability

.. |code_climate_maintainability| image:: https://api.codeclimate.com/v1/badges/2aa9313bbb9b0dc489cf/maintainability
    :target: https://codeclimate.com/github/LucaCappelletti94/covidxpert/maintainability
    :alt: Maintainability

.. |code_climate_coverage| image:: https://api.codeclimate.com/v1/badges/2aa9313bbb9b0dc489cf/test_coverage
    :target: https://codeclimate.com/github/LucaCappelletti94/covidxpert/test_coverage
    :alt: Code Climate Coverage
