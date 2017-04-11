import java.util.*;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class PlayerSkeleton {

	// CONSTANTS
	private final int NUM_WEIGHTS = 4;
	private final int NUM_POPULATION = 1000;
	private final int FITNESS_POPULATION = 100;
	private final int NUM_CHILDREN = 300;
	private final int NUM_MAX_MOVES = 500;
	private final int NUM_GAMES = 5;
	private final int NUM_ITERATIONS = 30;

	private class Weight {
		public double[] weights;
		public int score;

		public Weight(double[] weights) {
			this.weights = weights;
			score = 0;
		}
	}

	private class WeightComparator implements Comparator<Weight> {
		public int compare(Weight w1, Weight w2) {
			return w1.score - w2.score;
		}
		public boolean equals(Object obj) {
			return this == obj;
		}
	}

	private class EvaluateWeightTask implements Callable<Integer> {
		private Weight weight;

		public EvaluateWeightTask(Weight weight) {
			this.weight = weight;
		}

		@Override
	    public Integer call() throws Exception {
			ExecutorService executor = Executors.newFixedThreadPool(10);
			List<Future<Integer>> list = new ArrayList<Future<Integer>>();


			for (int g = 0; g < NUM_GAMES; g++) {
				Callable<Integer> callable = new PlayGameTask(weight);
				Future<Integer> future = executor.submit(callable);
				list.add(future);
			}

			int fitnessScore = 0;
			for(Future<Integer> fut : list){
            	try {
					fitnessScore += fut.get();
            	} catch (InterruptedException | ExecutionException e) {
                	e.printStackTrace();
            	}
        	}
			executor.shutdown();
			return fitnessScore;
	    }
	}

	private class PlayGameTask implements Callable<Integer> {
		private Weight weight;

		public PlayGameTask(Weight weight) {
			this.weight = weight;
		}

		@Override
	    public Integer call() throws Exception {
			return playForFitness(weight);
	    }

		private int playForFitness(Weight weight) {
			State s = new State();
			int numMoves = 0;

			while(!s.hasLost() && numMoves < NUM_MAX_MOVES) {
				s.makeMove(this.pickMoveFitness(s, s.legalMoves(), weight));
				numMoves++;
			}
			return s.getRowsCleared();
		}

		//implement this function to have a working system
		private int pickMoveFitness(State s, int[][] legalMoves, Weight weight) {
			int bestChoice = 0;
			double bestScore = Double.NEGATIVE_INFINITY;

			for (int choice = 0; choice < legalMoves.length; choice++) {
				int[][] field = new int[s.getField().length][];
				for (int i = 0; i < field.length; i++)
					field[i] = s.getField()[i].clone();

				int orient = legalMoves[choice][0];
				int slot = legalMoves[choice][1];

				// height if the first column makes contact
				int height = s.getTop()[slot] - State.getpBottom()[s.getNextPiece()][orient][0];
					// for each column beyond the first in the piece
				for (int c = 1; c < State.getpWidth()[s.getNextPiece()][orient]; c++) {
					height = Math.max(height, s.getTop()[slot + c] - State.getpBottom()[s.getNextPiece()][orient][c]);
				}

				// for each column in the piece - fill in the appropriate blocks
				for (int i = 0; i < State.getpWidth()[s.getNextPiece()][orient]; i++) {
					// from bottom to top of brick
					for (int h = height + State.getpBottom()[s.getNextPiece()][orient][i]; h < height
						 + State.getpTop()[s.getNextPiece()][orient][i]; h++) {
							 if (!(h >= field.length))
								field[h][i + slot] = s.getTurnNumber();
					}
				}

				double score = evaluate(weight.weights, field);
				if (score > bestScore) {
					bestScore = score;
					bestChoice = choice;
				}
			}
			return bestChoice;
		}
	}


	private double[] weights = new double[NUM_WEIGHTS];

	public static void main(String[] args) {
		PlayerSkeleton p = new PlayerSkeleton();
	}

	public PlayerSkeleton() {
		Weight w = geneticLearning();
		System.out.println(w.weights[0]);
		System.out.println(w.weights[1]);
		System.out.println(w.weights[2]);
		System.out.println(w.weights[3]);
	}

	//implement this function to have a working system
	public int pickMove(State s, int[][] legalMoves) {
		return 0;
	}

	/* FEATURES */
	private int aggregateHeight(int[][] field) {
		int aggregateHeight = 0;
		for (int c = 0; c < field[0].length; c++) {
			aggregateHeight += getColumnHeight(c, field);
		}
		return aggregateHeight;
	}

	private int numCompleteLines(int[][] field) {
		int numCompleteLines = 0;

		for (int r = 0; r < field.length; r++) {
			boolean complete = true;
			for (int c = 0; c < field[0].length; c++) {
				if (field[r][c] == 0) {
					complete = false;
					break;
				}
			}
			if (complete) numCompleteLines++;
		}
		return numCompleteLines;
	}

	private int numHoles(int[][] field) {
		int numHoles = 0;

		for (int c = 0; c < field[0].length; c++) {
			int columnHeight = getColumnHeight(c, field);
			for (int r = 0; r < columnHeight; r++) {
				if (field[r][c] == 0) numHoles++;
			}
		}
		return numHoles;
	}

	private int bumpiness(int[][] field) {
		int bumpiness = 0;

		for (int c = 0; c < field[0].length - 1; c++) {
			int columnHeightCurrent = getColumnHeight(c, field);
			int columnHeightNext = getColumnHeight(c + 1, field);
			bumpiness += Math.abs(columnHeightCurrent - columnHeightNext);
		}
		return bumpiness;
	}

	private int getColumnHeight(int col, int[][] field) {
		for (int i = field.length - 1; i >= 0; i--) {
			if (field[i][col] != 0) return i;
		}
		return 0;
	}

	/* SCORE FUNCTION */
	private double evaluate(double[] weights, int[][] field) {
		int aggregateHeight = aggregateHeight(field);
		int numCompleteLines = numCompleteLines(field);
		int numHoles = numHoles(field);
		int bumpiness = bumpiness(field);

		return weights[0] * aggregateHeight + weights[1] * numCompleteLines
			 + weights[2] * numHoles + weights[3] * bumpiness;
	}

	/* HELPER FUNCTIONS */
	private void printField(int[][] field) {
		for (int i = field.length - 1; i >= 0; i--) {
			for (int j = 0; j < field[0].length; j++) {
				System.out.print(field[i][j]);
			}
			System.out.println();
		}
		System.out.println();
	}



	/* GENETIC ALGORITHM */
	private Weight geneticLearning() {
		Weight[] population = new Weight[NUM_POPULATION];
		initPopulation(population);

		for (int i = 0; i < NUM_ITERATIONS; i++) {
			Weight[] childrenPopulation = new Weight[NUM_CHILDREN];

			for (int c = 0; c < childrenPopulation.length; c++) {
				System.out.println("I " + i + " C " + c);
				// Fitness for 10% of population
				Weight[] fitnessPopulation = new Weight[FITNESS_POPULATION];
				initFitnessPopulation(population, fitnessPopulation);

				runFitnessOnPopulation(fitnessPopulation);
				Arrays.sort(fitnessPopulation, new WeightComparator());

				//Perform crossover
				Weight crossover = weightedAvgCrossover(fitnessPopulation[fitnessPopulation.length - 1],
														fitnessPopulation[fitnessPopulation.length - 2]);
				//Perform mutation
				mutate(crossover);
				normalise(crossover);
				childrenPopulation[c] = crossover;
			}

			replaceLastN(population, childrenPopulation);
			clearPopulationScore(population);
		}

		runFitnessOnPopulation(population);
		Arrays.sort(population, new WeightComparator());
		System.out.println(population[population.length - 1].score);
		return population[population.length - 1];
	}

	// Evaluate fitness function
	private void runFitnessOnPopulation(Weight[] population) {
		ExecutorService executor = Executors.newFixedThreadPool(population.length);
		List<Future<Integer>> list = new ArrayList<Future<Integer>>();

		for (int w = 0; w < population.length; w++) {
			Callable<Integer> callable = new EvaluateWeightTask(population[w]);
			Future<Integer> future = executor.submit(callable);
			list.add(future);
		}

		for(int w = 0; w < list.size(); w++){
			try {
				population[w].score = list.get(w).get();
			} catch (InterruptedException | ExecutionException e) {
				e.printStackTrace();
			}
		}
		executor.shutdown();
	}

	private void initPopulation(Weight[] population) {
		for (int i=0; i < NUM_POPULATION; i++) {
			double[] weights = new double[NUM_WEIGHTS];
			weights[0] = -Math.random();
			weights[1] = Math.random();
			weights[2] = -Math.random();
			weights[3] = -Math.random();
			population[i] = new Weight(weights);
		}
	}

	private void initFitnessPopulation(Weight[] population, Weight[] fitnessPopulation) {
		Random r = new Random();
		for (int i=0; i < fitnessPopulation.length; i++) {
			fitnessPopulation[i] = population[r.nextInt(population.length)];
		}
	}

	private void clearPopulationScore(Weight[] population) {
		for (int w = 0; w < population.length; w++) {
			population[w].score = 0;
		}
	}

	private void replaceLastN(Weight[] population, Weight[] childrenPopulation) {
		runFitnessOnPopulation(population);
		Arrays.sort(population, new WeightComparator());

		for (int r = 0; r < childrenPopulation.length; r++) {
			population[r + population.length - childrenPopulation.length] = childrenPopulation[r];
		}
	}

	private Weight weightedAvgCrossover(Weight fittest, Weight secondFittest) {
		double[] crossover = new double[NUM_WEIGHTS];

		for (int w = 0; w < NUM_WEIGHTS; w++) {
			crossover[w] = fittest.weights[w] * fittest.score
						 + secondFittest.weights[w] * secondFittest.score;
		}
		return new Weight(crossover);
	}

	private void mutate(Weight crossover) {
		if(Math.random() < 0.05) {
			Random r = new Random();
			int component = r.nextInt(NUM_WEIGHTS);
			if (Math.random() < 0.5) {
				crossover.weights[component] *= 1.2;
			} else {
				crossover.weights[component] *= 0.8;
			}
		}
	}

	private void normalise(Weight crossover) {
		double total = 0;
		for (int i = 0; i < NUM_WEIGHTS; i++) {
			total += Math.pow(crossover.weights[i], 2);
		}
		total = Math.sqrt(total);

		for (int j = 0; j < NUM_WEIGHTS; j++) {
			crossover.weights[j] /= total;
		}
	}
}
