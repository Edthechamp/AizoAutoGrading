import pytest
import json
import cv2
import numpy as np
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'qt_app'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'source'))

from qt_app.checkAns import dataNormalize, GradeTest, load_answer_sheet
from source.process_image import extract_document, get_topic_boxes
import flaskApp
from flaskApp import app as flask_test_app


class TestDataNormalizeBoundaryValues:
    """Test boundary conditions for dataNormalize function"""
    
    def test_empty_student_sheet(self):
        """BV Test 1: Empty student sheet with valid answer sheet"""
        answer_sheet = {"1": "A", "2": "B", "3": "C"}
        student_sheet = {}
        
        result = dataNormalize(student_sheet, answer_sheet)
        
        assert result == {"1": None, "2": None, "3": None}
    
    def test_empty_answer_sheet(self):
        """BV Test 2: Empty answer sheet with student responses"""
        student_sheet = {"1": "A", "2": "B", "3": "C"}
        answer_sheet = {}
        
        result = dataNormalize(student_sheet, answer_sheet)
        
        assert result == {}
    
    def test_missing_student_answers(self):
        """BV Test 3: Student sheet missing some questions"""
        answer_sheet = {"1": "A", "2": "B", "3": "C", "4": "D"}
        student_sheet = {"1": "A", "3": "C"}
        
        result = dataNormalize(student_sheet, answer_sheet)
        
        assert result == {"1": "A", "2": None, "3": "C", "4": None}
    
    def test_non_numeric_question_keys(self):
        """BV Test 4: Non-numeric question keys"""
        answer_sheet = {"Q1": "A", "Q2": "B", "Q3": "C"}
        student_sheet = {"Q2": "B", "Q1": "B"}
        
        result = dataNormalize(student_sheet, answer_sheet)
        
        assert list(result.keys()) == ["Q1", "Q2", "Q3"]
        assert result["Q1"] == "B"
        assert result["Q2"] == "B"
        assert result["Q3"] is None
    
    def test_overflow_student_responses(self):
        """BV Test 5: Student answers more questions than expected"""
        answer_sheet = {"1": "A", "2": "B"}
        student_sheet = {"1": "A", "2": "B", "3": "D", "4": "A", "5": "C"}
        
        result = dataNormalize(student_sheet, answer_sheet)
        
        assert list(result.keys()) == ["1", "2"]
        assert result["1"] == "A"
        assert result["2"] == "B"
    
    def test_zero_valued_answers(self):
        """BV Test 6: Zero or falsy values in answers"""
        answer_sheet = {"1": "0", "2": "", "3": "A"}
        student_sheet = {"1": "0", "2": "", "3": "A"}
        
        result = dataNormalize(student_sheet, answer_sheet)
        
        assert result["1"] == "0"
        assert result["2"] == ""
        assert result["3"] == "A"


class TestGradeTestBoundaryValues:
    """Test boundary conditions for GradeTest function"""
    
    @patch('qt_app.checkAns.load_answer_sheet')
    def test_all_incorrect_answers(self, mock_load):
        """BV Test 7: Student answers all questions incorrectly"""
        mock_load.return_value = {
            "math": {"1": "A", "2": "B", "3": "C"}
        }
        
        test = {
            "studentID": "STU001",
            "answers": {
                "math": {"1": "B", "2": "C", "3": "A"}
            }
        }
        
        result = GradeTest(test)
        
        assert result["code"] == "STU001"
        assert result["scores"]["math"] == 0
    
    @patch('qt_app.checkAns.load_answer_sheet')
    def test_all_correct_answers(self, mock_load):
        """BV Test 8: Student answers all questions correctly"""
        mock_load.return_value = {
            "science": {"1": "A", "2": "B"}
        }
        
        test = {
            "studentID": "STU002",
            "answers": {
                "science": {"1": "A", "2": "B"}
            }
        }
        
        result = GradeTest(test)
        
        assert result["code"] == "STU002"
        assert result["scores"]["science"] == 2
    
    @patch('qt_app.checkAns.load_answer_sheet')
    def test_missing_entire_section(self, mock_load):
        """BV Test 9: Student section with missing questions triggers normalization"""
        mock_load.return_value = {
            "section1": {"1": "A", "2": "B", "3": "C"},
            "section2": {"1": "C", "2": "D"}
        }
        
        test = {
            "studentID": "STU003",
            "answers": {
                "section1": {"1": "A", "2": "B", "3": "C"},
                "section2": {"1": "C"}
            }
        }
        
        result = GradeTest(test)
        
        assert result["code"] == "STU003"
        assert result["scores"]["section1"] == 3
        assert result["scores"]["section2"] == 1


class TestImageProcessingBoundaryValues:
    """Test boundary conditions for image processing functions"""
    
    def test_extract_document_minimal_image(self):
        """BV Test 10: Minimal size image (50x50)"""
        image = np.ones((50, 50, 3), dtype=np.uint8) * 255
        
        result = extract_document(image)
        
        assert result is not None
        assert isinstance(result, np.ndarray)
    
    def test_get_topic_boxes_blank_image(self):
        """BV Test 11: Blank/empty image"""
        image = np.ones((400, 400, 3), dtype=np.uint8) * 255
        
        result = get_topic_boxes(image)
        
        assert isinstance(result, list)
    
    def test_get_topic_boxes_single_box(self):
        """BV Test 12: Image with single bounding box"""
        image = np.ones((400, 400, 3), dtype=np.uint8) * 255
        cv2.rectangle(image, (50, 50), (350, 350), (0, 0, 0), -1)
        
        result = get_topic_boxes(image, debug=False)
        
        assert isinstance(result, list)

class TestFullGradingWorkflow:
    """Integration tests for complete grading pipeline"""
    
    @patch('qt_app.checkAns.load_answer_sheet')
    def test_integration_extract_normalize_grade(self, mock_load):
        """Integration Test 1: Full pipeline - normalize answers and grade"""
        mock_load.return_value = {
            "section_A": {"1": "A", "2": "B", "3": "C"},
            "section_B": {"1": "D", "2": "A", "3": "B"}
        }
        
        test_data = {
            "studentID": "STU12345",
            "answers": {
                "section_A": {"1": "A", "3": "C"},
                "section_B": {"2": "A", "1": "D", "3": "B"}
            }
        }
        
        normalized_A = dataNormalize(
            test_data["answers"]["section_A"],
            mock_load.return_value["section_A"]
        )
        normalized_B = dataNormalize(
            test_data["answers"]["section_B"],
            mock_load.return_value["section_B"]
        )
        
        result = GradeTest(test_data)
        
        assert result["code"] == "STU12345"
        assert "section_A" in result["scores"]
        assert "section_B" in result["scores"]
        assert result["scores"]["section_A"] == 2
        assert result["scores"]["section_B"] == 3
    
    def test_integration_image_document_extraction(self):
        """Integration Test 2: Image processing pipeline - extract and detect boxes"""
        image = np.ones((600, 800, 3), dtype=np.uint8) * 240
        
        cv2.rectangle(image, (50, 50), (750, 550), (0, 0, 0), 3)
        
        cv2.rectangle(image, (100, 100), (350, 200), (50, 50, 50), -1)
        cv2.rectangle(image, (450, 100), (700, 200), (50, 50, 50), -1)
        cv2.rectangle(image, (100, 250), (700, 480), (60, 60, 60), -1)
        
        extracted = extract_document(image, debug=False)
        
        assert extracted is not None
        assert isinstance(extracted, np.ndarray)
        
        boxes = get_topic_boxes(extracted, debug=False)
        
        assert isinstance(boxes, list)
        assert len(boxes) >= 0

    def test_integration_img_upload_get_form(self):
        """Integration Test 3: Flask upload endpoint GET returns upload form"""
        client = flask_test_app.test_client()

        response = client.get("/imgUpload")

        assert response.status_code == 200
        assert b"<form method=\"POST\" action=\"/imgUpload\"" in response.data

    def test_integration_img_upload_post_success(self, tmp_path, monkeypatch):
        """Integration Test 4: Flask upload endpoint POST saves file and returns JSON"""
        monkeypatch.setattr(flaskApp, "UPLOAD_DIR", str(tmp_path))
        os.makedirs(flaskApp.UPLOAD_DIR, exist_ok=True)

        client = flask_test_app.test_client()
        data = {
            "image": (BytesIO(b"fake image data"), "student 01.png")
        }

        response = client.post("/imgUpload", data=data, content_type="multipart/form-data")

        assert response.status_code == 200
        payload = response.get_json()
        assert payload["message"] == "Upload successful"
        assert payload["filename"] == "student_01.png"
        assert (tmp_path / payload["filename"]).exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
