#!/usr/bin/env python3
"""
Evaluation script for GeoLingua project.
Evaluates the trained model on various benchmarks and metrics.
"""

import os
import sys
import argparse
import logging
import json
import torch
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.base_model import GeoLinguaModel
from evaluation.metrics import GeoLinguaEvaluator
from evaluation.benchmarks import GeographicBenchmark
from data.loaders import GeoLinguaDataLoader
from config.model_config import *
from config.data_config import REGIONS, PROCESSED_DATA_PATH


def setup_logging(log_level: str = 'INFO', log_file: str = None):
    """Setup logging configuration."""
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def load_model(model_path: str, device: str) -> GeoLinguaModel:
    """
    Load trained GeoLingua model.
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load model on
    
    Returns:
        Loaded GeoLingua model
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading model from: {model_path}")
    
    # Load model configuration
    config_file = os.path.join(os.path.dirname(model_path), 'training_config.json')
    
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        model_name = config.get('model_name', MODEL_NAME)
        regions = config.get('regions', list(REGIONS.keys()))
    else:
        logger.warning(f"Config file not found: {config_file}")
        logger.warning("Using default configuration")
        model_name = MODEL_NAME
        regions = list(REGIONS.keys())
    
    # Initialize model
    model = GeoLinguaModel(
        model_name=model_name,
        regions=regions,
        lora_config={
            'r': LORA_R,
            'lora_alpha': LORA_ALPHA,
            'lora_dropout': LORA_DROPOUT,
            'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        }
    )
    
    # Load checkpoint
    try:
        if os.path.isfile(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Try to load from directory
            model.load_pretrained(model_path)
        
        model = model.to(device)
        model.eval()
        
        logger.info(f"Model loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def load_evaluation_data(data_dir: str, regions: List[str]) -> Dict:
    """
    Load evaluation data.
    
    Args:
        data_dir: Directory containing processed data
        regions: List of regions to evaluate
    
    Returns:
        Dictionary containing evaluation datasets
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f"Loading evaluation data for regions: {regions}")
    
    # Load processed data
    processed_file = os.path.join(data_dir, "processed_data.json")
    
    if not os.path.exists(processed_file):
        raise FileNotFoundError(f"Processed data not found: {processed_file}")
    
    with open(processed_file, 'r') as f:
        processed_data = json.load(f)
    
    # Initialize data loader
    data_loader = GeoLinguaDataLoader(
        processed_data=processed_data,
        batch_size=1,  # Use batch size 1 for evaluation
        max_length=MAX_LENGTH,
        regions=regions
    )
    
    # Get evaluation datasets
    eval_datasets = data_loader.get_evaluation_datasets()
    
    logger.info(f"Loaded evaluation data for {len(eval_datasets)} regions")
    return eval_datasets


def run_geographic_evaluation(model: GeoLinguaModel, eval_datasets: Dict, 
                            device: str, output_dir: str) -> Dict:
    """
    Run geographic evaluation on the model.
    
    Args:
        model: Trained GeoLingua model
        eval_datasets: Dictionary of evaluation datasets by region
        device: Device to run evaluation on
        output_dir: Output directory for results
    
    Returns:
        Dictionary containing evaluation results
    """
    logger = logging.getLogger(__name__)
    
    logger.info("Starting geographic evaluation")
    
    # Initialize evaluator
    evaluator = GeoLinguaEvaluator()
    
    # Initialize benchmarks
    benchmark = GeographicBenchmark()
    
    results = {}
    
    for region_name, dataset in eval_datasets.items():
        logger.info(f"Evaluating region: {region_name}")
        
        region_results = {
            'region': region_name,
            'samples_evaluated': 0,
            'metrics': {}
        }
        
        # Collect predictions and references
        predictions = []
        references = []
        contexts = []
        
        # Process evaluation samples
        for batch in tqdm(dataset, desc=f"Evaluating {region_name}"):
            try:
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                region_id = batch['region_id'].to(device)
                
                # Generate predictions
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        region_id=region_id,
                        max_length=MAX_LENGTH,
                        num_return_sequences=1,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=model.tokenizer.pad_token_id
                    )
                
                # Decode predictions
                prediction = model.tokenizer.decode(
                    outputs[0], skip_special_tokens=True
                )
                
                # Get reference text
                reference = batch.get('target_text', [''])[0]
                context = batch.get('context', [''])[0]
                
                predictions.append(prediction)
                references.append(reference)
                contexts.append(context)
                
                region_results['samples_evaluated'] += 1
                
            except Exception as e:
                logger.error(f"Error processing batch in {region_name}: {e}")
                continue
        
        # Calculate metrics
        if predictions and references:
            # Language quality metrics
            bleu_score = evaluator.calculate_bleu(predictions, references)
            rouge_scores = evaluator.calculate_rouge(predictions, references)
            perplexity = evaluator.calculate_perplexity(model, predictions, device)
            
            # Geographic appropriateness metrics
            geographic_score = evaluator.calculate_geographic_appropriateness(
                predictions, contexts, region_name
            )
            
            # Cultural context metrics
            cultural_score = evaluator.calculate_cultural_context_score(
                predictions, region_name
            )
            
            # Linguistic pattern metrics
            linguistic_score = evaluator.calculate_linguistic_pattern_score(
                predictions, region_name
            )
            
            # Store metrics
            region_results['metrics'] = {
                'bleu': bleu_score,
                'rouge': rouge_scores,
                'perplexity': perplexity,
                'geographic_appropriateness': geographic_score,
                'cultural_context': cultural_score,
                'linguistic_patterns': linguistic_score
            }
            
            logger.info(f"Region {region_name} - BLEU: {bleu_score:.4f}, "
                       f"Geographic: {geographic_score:.4f}, "
                       f"Cultural: {cultural_score:.4f}")
        
        results[region_name] = region_results
        
        # Save individual region results
        region_output_file = os.path.join(output_dir, f"{region_name}_results.json")
        with open(region_output_file, 'w') as f:
            json.dump(region_results, f, indent=2)
    
    logger.info("Geographic evaluation completed")
    return results


def run_benchmark_evaluation(model: GeoLinguaModel, device: str, 
                           output_dir: str) -> Dict:
    """
    Run benchmark evaluation on standard geographic tasks.
    
    Args:
        model: Trained GeoLingua model
        device: Device to run evaluation on
        output_dir: Output directory for results
    
    Returns:
        Dictionary containing benchmark results
    """
    logger = logging.getLogger(__name__)
    
    logger.info("Starting benchmark evaluation")
    
    # Initialize benchmark
    benchmark = GeographicBenchmark()
    
    # Get benchmark tasks
    benchmark_tasks = benchmark.get_all_tasks()
    
    results = {}
    
    for task_name, task_data in benchmark_tasks.items():
        logger.info(f"Running benchmark: {task_name}")
        
        task_results = {
            'task': task_name,
            'samples': len(task_data),
            'accuracy': 0.0,
            'predictions': []
        }
        
        correct_predictions = 0
        
        for sample in tqdm(task_data, desc=f"Benchmark {task_name}"):
            try:
                # Prepare input
                input_text = sample['input']
                expected_output = sample['output']
                region = sample.get('region', 'global')
                
                # Tokenize input
                inputs = model.tokenizer(
                    input_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=MAX_LENGTH
                )
                
                # Move to device
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs['attention_mask'].to(device)
                
                # Get region ID
                region_id = torch.tensor([model.region_to_id.get(region, 0)]).to(device)
                
                # Generate prediction
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        region_id=region_id,
                        max_length=MAX_LENGTH,
                        num_return_sequences=1,
                        do_sample=False,
                        pad_token_id=model.tokenizer.pad_token_id
                    )
                
                # Decode prediction
                prediction = model.tokenizer.decode(
                    outputs[0], skip_special_tokens=True
                )
                
                # Check if prediction matches expected output
                if benchmark.evaluate_prediction(prediction, expected_output, task_name):
                    correct_predictions += 1
                
                task_results['predictions'].append({
                    'input': input_text,
                    'prediction': prediction,
                    'expected': expected_output,
                    'correct': benchmark.evaluate_prediction(prediction, expected_output, task_name)
                })
                
            except Exception as e:
                logger.error(f"Error processing benchmark sample: {e}")
                continue
        
        # Calculate accuracy
        task_results['accuracy'] = correct_predictions / len(task_data) if task_data else 0.0
        
        logger.info(f"Benchmark {task_name} - Accuracy: {task_results['accuracy']:.4f}")
        
        results[task_name] = task_results
        
        # Save individual benchmark results
        benchmark_output_file = os.path.join(output_dir, f"benchmark_{task_name}.json")
        with open(benchmark_output_file, 'w') as f:
            json.dump(task_results, f, indent=2)
    
    logger.info("Benchmark evaluation completed")
    return results


def generate_evaluation_report(geographic_results: Dict, benchmark_results: Dict, 
                             output_dir: str) -> str:
    """
    Generate comprehensive evaluation report.
    
    Args:
        geographic_results: Results from geographic evaluation
        benchmark_results: Results from benchmark evaluation
        output_dir: Output directory for report
    
    Returns:
        Path to generated report
    """
    logger = logging.getLogger(__name__)
    
    logger.info("Generating evaluation report")
    
    # Create report
    report = {
        'evaluation_timestamp': datetime.now().isoformat(),
        'summary': {
            'total_regions_evaluated': len(geographic_results),
            'total_benchmarks_evaluated': len(benchmark_results),
            'overall_performance': {}
        },
        'geographic_evaluation': geographic_results,
        'benchmark_evaluation': benchmark_results,
        'detailed_analysis': {}
    }
    
    # Calculate overall performance metrics
    if geographic_results:
        # Average metrics across regions
        avg_bleu = np.mean([r['metrics'].get('bleu', 0) for r in geographic_results.values()])
        avg_geographic = np.mean([r['metrics'].get('geographic_appropriateness', 0) 
                                 for r in geographic_results.values()])
        avg_cultural = np.mean([r['metrics'].get('cultural_context', 0) 
                               for r in geographic_results.values()])
        avg_linguistic = np.mean([r['metrics'].get('linguistic_patterns', 0) 
                                 for r in geographic_results.values()])
        
        report['summary']['overall_performance']['average_bleu'] = avg_bleu
        report['summary']['overall_performance']['average_geographic_appropriateness'] = avg_geographic
        report['summary']['overall_performance']['average_cultural_context'] = avg_cultural
        report['summary']['overall_performance']['average_linguistic_patterns'] = avg_linguistic
    
    if benchmark_results:
        # Average benchmark accuracy
        avg_accuracy = np.mean([r['accuracy'] for r in benchmark_results.values()])
        report['summary']['overall_performance']['average_benchmark_accuracy'] = avg_accuracy
    
    # Detailed analysis
    report['detailed_analysis']['best_performing_region'] = max(
        geographic_results.items(),
        key=lambda x: x[1]['metrics'].get('geographic_appropriateness', 0)
    )[0] if geographic_results else None
    
    report['detailed_analysis']['best_performing_benchmark'] = max(
        benchmark_results.items(),
        key=lambda x: x[1]['accuracy']
    )[0] if benchmark_results else None
    
    # Save report
    report_file = os.path.join(output_dir, 'evaluation_report.json')
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Generate human-readable summary
    summary_file = os.path.join(output_dir, 'evaluation_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("GeoLingua Model Evaluation Summary\n")
        f.write("=" * 40 + "\n\n")
        
        f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("Geographic Evaluation:\n")
        f.write(f"  Regions Evaluated: {len(geographic_results)}\n")
        if geographic_results:
            f.write(f"  Average BLEU Score: {avg_bleu:.4f}\n")
            f.write(f"  Average Geographic Appropriateness: {avg_geographic:.4f}\n")
            f.write(f"  Average Cultural Context: {avg_cultural:.4f}\n")
            f.write(f"  Average Linguistic Patterns: {avg_linguistic:.4f}\n")
        f.write("\n")
        
        f.write("Benchmark Evaluation:\n")
        f.write(f"  Benchmarks Evaluated: {len(benchmark_results)}\n")
        if benchmark_results:
            f.write(f"  Average Accuracy: {avg_accuracy:.4f}\n")
        f.write("\n")
        
        f.write("Regional Performance:\n")
        for region_name, results in geographic_results.items():
            metrics = results['metrics']
            f.write(f"  {region_name}:\n")
            f.write(f"    BLEU: {metrics.get('bleu', 0):.4f}\n")
            f.write(f"    Geographic: {metrics.get('geographic_appropriateness', 0):.4f}\n")
            f.write(f"    Cultural: {metrics.get('cultural_context', 0):.4f}\n")
            f.write(f"    Linguistic: {metrics.get('linguistic_patterns', 0):.4f}\n")
        f.write("\n")
        
        f.write("Benchmark Performance:\n")
        for benchmark_name, results in benchmark_results.items():
            f.write(f"  {benchmark_name}: {results['accuracy']:.4f}\n")
    
    logger.info(f"Evaluation report saved to: {report_file}")
    logger.info(f"Evaluation summary saved to: {summary_file}")
    
    return report_file


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate GeoLingua model')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, default=PROCESSED_DATA_PATH,
                       help='Directory containing processed data')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Output directory for evaluation results')
    parser.add_argument('--regions', type=str, nargs='+', default=None,
                       help='Specific regions to evaluate (default: all)')
    parser.add_argument('--skip_geographic', action='store_true',
                       help='Skip geographic evaluation')
    parser.add_argument('--skip_benchmark', action='store_true',
                       help='Skip benchmark evaluation')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--log_level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(args.output_dir, 'evaluation.log')
    setup_logging(args.log_level, log_file)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting GeoLingua evaluation")
    
    # Determine device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    try:
        # Load model
        model = load_model(args.model_path, device)
        
        # Determine regions to evaluate
        if args.regions:
            regions = args.regions
        else:
            regions = list(REGIONS.keys())
        
        logger.info(f"Evaluating regions: {regions}")
        
        # Initialize results
        geographic_results = {}
        benchmark_results = {}
        
        # Run geographic evaluation
        if not args.skip_geographic:
            eval_datasets = load_evaluation_data(args.data_dir, regions)
            geographic_results = run_geographic_evaluation(
                model, eval_datasets, device, args.output_dir
            )
        
        # Run benchmark evaluation
        if not args.skip_benchmark:
            benchmark_results = run_benchmark_evaluation(
                model, device, args.output_dir
            )
        
        # Generate evaluation report
        report_file = generate_evaluation_report(
            geographic_results, benchmark_results, args.output_dir
        )
        
        logger.info(f"Evaluation completed successfully")
        logger.info(f"Results saved to: {args.output_dir}")
        logger.info(f"Report saved to: {report_file}")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()
