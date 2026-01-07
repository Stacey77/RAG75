"""
Test Production RAG Pipeline Setup
Validates the production deployment before going live
"""

import os
import sys
import subprocess
import time
import requests
from pathlib import Path

def check_prerequisites():
    """Check if all prerequisites are met"""
    print("Checking Prerequisites...")
    
    # Check Python
    python_version = sys.version_info
    print(f"   Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    # Check required files
    required_files = [
        "rag_pipeline_advanced.py",
        "README.md",
        "sample.txt",
        "production/app.py",
        "production/requirements.txt",
        "production/Dockerfile",
        "production/docker-compose.yml"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"   ❌ Missing files: {missing_files}")
        return False
    else:
        print("   All required files present")
    
    # Check Docker (if available)
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        print(f"   Docker: {result.stdout.strip()}")
    except FileNotFoundError:
        print("   Docker not found (local deployment only)")
    
    return True

def test_imports():
    """Test if all required modules can be imported"""
    print("\nTesting Imports...")
    
    try:
        # Test core RAG pipeline
        sys.path.insert(0, ".")
        from rag_pipeline_advanced import RAGPipeline
        print("   RAG Pipeline import successful")
        
        # Test FastAPI (production)
        try:
            from fastapi import FastAPI
            print("   FastAPI import successful")
        except ImportError:
            print("   FastAPI not available (install with: pip install fastapi)")
        
        # Test other dependencies
        try:
            import numpy
            print("   NumPy available")
        except ImportError:
            print("   NumPy missing")
        
        try:
            import sentence_transformers
            print("   Sentence Transformers available")
        except ImportError:
            print("   Sentence Transformers missing")
        
        try:
            import faiss
            print("   FAISS available")
        except ImportError:
            print("   FAISS missing")
        
        return True
        
    except ImportError as e:
        print(f"   Import failed: {e}")
        return False

def test_pipeline_functionality():
    """Test basic pipeline functionality"""
    print("\nTesting Pipeline Functionality...")
    
    try:
        from rag_pipeline_advanced import RAGPipeline
        
        # Initialize pipeline
        pipeline = RAGPipeline(chunk_size=400, top_k=3, rerank_k=2)
        print("   Pipeline initialization successful")
        
        # Add sources
        sources = ["README.md", "sample.txt"]
        if all(os.path.exists(s) for s in sources):
            pipeline.add_sources(sources)
            print(f"   Sources added: {len(sources)} files")
        else:
            print("   Some source files missing")
        
        # Test query
        query = "What is this about?"
        result = pipeline.query(query)
        print(f"   Query processed: {len(result)} chars response")
        
        # Test retrieval
        context = pipeline.retrieve(query)
        print(f"   Content retrieval: {len(context)} chunks")
        
        return True
        
    except Exception as e:
        print(f"   Pipeline test failed: {e}")
        return False

def test_production_app():
    """Test production app structure"""
    print("\nTesting Production App...")
    
    try:
        # Check if app.py can be imported
        sys.path.insert(0, "production")
        import app
        print("   Production app import successful")
        
        # Check FastAPI app
        if hasattr(app, 'app') and hasattr(app.app, 'title'):
            print(f"   FastAPI app: {app.app.title}")
        
        return True
        
    except ImportError as e:
        print(f"   Production app import failed: {e}")
        print("   (This is expected if FastAPI is not installed)")
        return False
    except Exception as e:
        print(f"   Production app test failed: {e}")
        return False

def test_configuration():
    """Test configuration files"""
    print("\nTesting Configuration...")
    
    # Check production requirements
    req_file = "production/requirements.txt"
    if os.path.exists(req_file):
        with open(req_file, 'r') as f:
            requirements = f.read().strip().split('\n')
        print(f"   Production requirements: {len(requirements)} packages")
    
    # Check Docker compose
    compose_file = "production/docker-compose.yml"
    if os.path.exists(compose_file):
        print("   Docker Compose configuration present")
    
    # Check environment example
    env_file = "production/.env.example"
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            env_vars = len([line for line in f if '=' in line and not line.startswith('#')])
        print(f"   Environment variables: {env_vars} configured")
    
    return True

def generate_deployment_report():
    """Generate deployment readiness report"""
    print("\nGenerating Deployment Report...")
    
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "files_present": [],
        "tests_passed": [],
        "recommendations": []
    }
    
    # Check files
    required_files = [
        "rag_pipeline_advanced.py",
        "production/app.py",
        "production/requirements.txt",
        "production/Dockerfile",
        "production/docker-compose.yml",
        "production/.env.example"
    ]
    
    for file in required_files:
        if os.path.exists(file):
            report["files_present"].append(file)
    
    # Run tests
    tests = [
        ("Prerequisites", check_prerequisites),
        ("Imports", test_imports),
        ("Pipeline", test_pipeline_functionality),
        ("Production App", test_production_app),
        ("Configuration", test_configuration)
    ]
    
    for test_name, test_func in tests:
        try:
            if test_func():
                report["tests_passed"].append(test_name)
        except Exception as e:
            print(f"   Test {test_name} failed: {e}")
    
    # Recommendations
    if len(report["tests_passed"]) >= 3:
        report["recommendations"].append("✅ Ready for local deployment")
    
    if "FastAPI" not in str(report.get("tests_passed", [])):
        report["recommendations"].append("⚠️  Install FastAPI for production API")
    
    if len(report["files_present"]) >= 5:
        report["recommendations"].append("✅ Production files complete")
    
    # Save report
    import json
    with open("deployment_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"   Report saved to deployment_report.json")
    return report

def main():
    """Main test function"""
    print("Production RAG Pipeline - Deployment Test")
    print("=" * 60)
    
    # Run all tests
    tests_passed = 0
    total_tests = 5
    
    if check_prerequisites():
        tests_passed += 1
    
    if test_imports():
        tests_passed += 1
    
    if test_pipeline_functionality():
        tests_passed += 1
    
    if test_production_app():
        tests_passed += 1
    
    if test_configuration():
        tests_passed += 1
    
    # Generate report
    report = generate_deployment_report()
    
    # Summary
    print("\n" + "=" * 60)
    print("DEPLOYMENT TEST SUMMARY")
    print("=" * 60)
    print(f"Tests Passed: {tests_passed}/{total_tests}")
    print(f"Files Ready: {len(report['files_present'])}/6")
    
    if tests_passed >= 4:
        print("\nREADY FOR PRODUCTION!")
        print("\nNext steps:")
        print("1. Install production dependencies: pip install -r production/requirements.txt")
        print("2. Set environment variables: cp production/.env.example production/.env")
        print("3. Start production server: python production/app.py")
        print("4. Or use Docker: cd production && docker-compose up -d")
    else:
        print("\nNOT READY FOR PRODUCTION")
        print("Please address the failed tests above.")
    
    print(f"\nDetailed report: deployment_report.json")

if __name__ == "__main__":
    main()
