package distances;

import writables.Point;

public class EuclideanDistance extends Distance {
    @Override
    public double getDistance(Point p1, Point p2) throws Exception {
        double[] p1Vector = p1.getVector();
        double[] p2Vector = p2.getVector();
    
        if (p1Vector.length != p2Vector.length) throw new Exception("Invalid length");
    
        double sum = 0;
        for (int i = 0; i < p1Vector.length; i++) {
            sum += Math.pow(p1Vector[i] - p2Vector[i], 2);
        }
    
        return sum; 
    }


    @Override
    public Point getExpectation(Iterable<Point> points) {
        Point result = sumPoints(points);

        if (result != null) {
            result.compress();
        }
        return result;
    }
}

public class CosineDistance extends Distance {
    @Override
    public double getDistance(Point p1, Point p2) throws Exception {
        double[] p1Vector = p1.getVector();
        double[] p2Vector = p2.getVector();

        if (p1Vector.length != p2Vector.length) throw new Exception("Invalid length");

        double dotProduct = 0, normP1 = 0, normP2 = 0;
        for (int i = 0; i < p1Vector.length; i++) {
            dotProduct += p1Vector[i] * p2Vector[i];
            normP1 += Math.pow(p1Vector[i], 2);
            normP2 += Math.pow(p2Vector[i], 2);
        }

        return 1 - (dotProduct / (Math.sqrt(normP1) * Math.sqrt(normP2)));
    }

    @Override
    public Point getExpectation(Iterable<Point> points) {
        return super.getExpectation(points); 
    }
}

