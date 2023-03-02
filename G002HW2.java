import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;


public class G002HW2
{
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // Input reading methods (provided)
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    /**
     * Transforms a string representation of a Vector into a spark.mllib.linalg.Vector
     *
     * @param str comma-separated doubles e.g.: "1, 4.8, 12, -15.588"
     * @return Vector representation of the string
     */
    public static Vector strToVector(String str)
    {
        String[] tokens = str.split(",");
        double[] data = new double[tokens.length];
        for (int i=0; i<tokens.length; i++)
        {
            data[i] = Double.parseDouble(tokens[i]);
        }
        return Vectors.dense(data); //transforms an array x of double into an instance of class Vector
    }

    /**
     * Reads a list of Vectors from a properly formatted file.
     * Each line must represent a Vector in the format used in strToVector()
     *
     * @param filename path to the file
     * @return
     * @throws IOException in case the file does not exist/is not readable
     * @throws IllegalArgumentException if filename represents a directory
     */
    public static ArrayList<Vector> readVectorsSeq(String filename) throws IOException
    {
        if (Files.isDirectory(Paths.get(filename)))
        {
            throw new IllegalArgumentException("readVectorsSeq is meant to read a single file.");
        }
        ArrayList<Vector> result = new ArrayList<>();
        Files.lines(Paths.get(filename))
                .map(str -> strToVector(str))
                .forEach(e -> result.add(e));
        return result;
    }


    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
    // Begin our code
    // &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

    /**
     * Core function that implements the k-center algorithm with outliers.
     *
     *
     * @param P input set of N points
     * @param W set of weights
     * @param k number of centers
     * @param z number of allowed outliers
     * @param alpha additional parameter used to a better evaluation of the Bz
     * @return Set S ⊂ P of k centers which minimize
     *              Φkcenter(P − Zs , S),
     *          where Zs = z points of P farthest from S.
     */
    public static ArrayList<Vector>SeqWeightedOutliers(ArrayList<Vector> P, ArrayList<Long> W, int k, int z, int alpha)
    {
        int n_iter=0;
        /*
         * Precompute distances for faster execution of Bz
         */
        double[][] distances = new double[P.size()][P.size()];
        for(int i=0; i<P.size(); i++)
        {
            for(int j=0; j<P.size(); j++)
            {
                distances[i][j] = Math.sqrt(Vectors.sqdist(P.get(i), P.get(j)));
            }
        }
        /*
         * Compute the initial value of r.
         * r is defined as the minimum distance between the first k+z+1 points in P divided by 2.
         */
        double raggio = Double.POSITIVE_INFINITY;
        for (int i=0; i<k+z+1; i++)
        {
            for (int j=i+1; j<k+z+1; j++)
            {
                double sqr_dist = Vectors.sqdist(P.get(i), P.get(j));
                if (raggio > sqr_dist)
                {
                    raggio = sqr_dist;
                }
            }
        }
        raggio = Math.sqrt(raggio)/2;
        System.out.println("Initial guess = " + raggio + " ");
        /*
         * Main loop
         */
        while(true)
        {
            /*
             * Initialize Wz (cumulative weight of Z) and Z (set of points yet to be covered)
             */
            long Wz = 0L;
            n_iter++;

            // Z <- P
            ArrayList<Integer> Z = new ArrayList<Integer>(P.size()); //just indices
            for(int i=0; i<P.size(); i++)
            {
                Z.add(i); //initialize indices
            }

            // S <- {}
            ArrayList<Vector> S = new ArrayList<Vector>();

            // Wz <- sum of weights in P
            for(long l : W)
            {
                Wz += l;
            }

            // beginning centers search
            while((S.size() < k) && (Wz > 0L))
            {
                long max = 0L;
                int idx_of_newcenter = -1;
                for(int i=0; i<P.size(); i++)
                {
                    // initialize Bz and keep track of its weight
                    ArrayList<Integer> Bz = new ArrayList<Integer>();
                    long ball_weight =0L;

                    for (int j : Z) //for each index in Z
                    {
                        if (distances[i][j]<=((1+2*alpha)*raggio)) //if the distance between point i and point of said index is <=(1+2alpha)r
                        {
                            Bz.add(j); //add the index to Bz
                        }
                    }
                    // adjust compute weight of the points in the ball
                    for(int index : Bz)
                    {
                        ball_weight += W.get(index);
                    }
                    if(ball_weight > max)
                    {
                        max = ball_weight;
                        idx_of_newcenter = i;
                    }
                }

                //add the new center
                S.add(P.get(idx_of_newcenter));

                // initialize Bz as list of indices
                ArrayList<Integer> Bz = new ArrayList<Integer>();
                for (int j : Z)
                {
                    if (distances[idx_of_newcenter][j]<=((3+4*alpha)*raggio))
                    {
                        Bz.add(j);
                    }
                }
                // remove Bz from Z
                ArrayList<Integer> to_remove = new ArrayList<>();
                for(int index : Bz)
                {
                    to_remove.add(index);
                    Wz -= W.get(index);
                }
                Z.removeAll(to_remove); //O(|Z|)
            }
            //loop end factor
            if(Wz <= z)
            {
                System.out.println("Final guess = " + raggio + " ");
                System.out.println("Number of guesses = " + n_iter + " ");
                return S;
            }
            //continue the loop
            else
            {
                raggio = 2*raggio;
            }
        }
    }

    /**
     * Computes the value of the objective function for the set of points P, the set of centers S, and z outliers
     * (the number of centers, which is the size of S, is not needed as a parameter)
     *
     * @param P input set of N points
     * @param S the output of the function SeqWeightedOutliers
     * @param z number of allowed outliers
     * @return objective function
     */

    public static double ComputeObjective(ArrayList<Vector> P, ArrayList<Vector> S, int z)
    {
        ArrayList<Double> sq_distances = new ArrayList<>(); //1 distance for each point
        for(Vector point : P) //O(N)
        {
            double min_dist = Double.POSITIVE_INFINITY; //minimum distance from point to a center
            for(Vector center : S) //O(k)
            {
                double dist = Vectors.sqdist(point, center);
                if(dist < min_dist)
                {
                    min_dist = dist;
                }
                if(min_dist == 0)
                {
                    break; //we cannot do better
                }
            }
            sq_distances.add(min_dist);
        }

        sq_distances.sort(Comparator.reverseOrder()); //sort from largest to smallest | O(N*logN)

        return Math.sqrt(sq_distances.get(z)); //return the square root of the (z+1)th largest element | O(1)
    }

    public static void main(String[] args) throws IOException
    {
        //cli arguments parsing
        if (args.length != 3)
        {
            throw new IllegalArgumentException(
                    "You must input the path to the input file, " +
                            "Z (the number of allowed outliers), " +
                            "and K (the number of centers)");
        }
        String str=args[0];
        int k = Integer.parseInt(args[1]);
        int z = Integer.parseInt(args[2]);

        //initialization of data structures from data
        ArrayList<Vector> P = readVectorsSeq(str);
        ArrayList<Long> weights = new ArrayList<>();
        for(int i=0;i<P.size();i++) //initialize weights (all weights) to 1
        {
            weights.add(1L);
        }

        System.out.println("Input size n = " + P.size() + " ");
        System.out.println("Number of centers k = " + k + " ");
        System.out.println("Number of outliers z = " + z + " ");

        //begin actual algorithm
        long startTime = System.currentTimeMillis();
        ArrayList<Vector> solution = SeqWeightedOutliers(P, weights, k, z, 0);
        long endTime = System.currentTimeMillis();
        double objective = ComputeObjective(P,solution,z);

        System.out.println("Objective function = " + objective + " ");
        System.out.println("Time of SeqWeightedOutliers = "+ (endTime-startTime));
    }
}