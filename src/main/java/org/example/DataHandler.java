package org.example;
import java.sql.*;
import java.util.*;
public class DataHandler {
    private static final String URL = "jdbc:mysql://localhost:3306/chatbot_db"; // Modified database name
    private static final String USER = "root";
    private static final String PASSWORD = "Jojo!007";
    public Connection getConnection() throws SQLException {
        return DriverManager.getConnection(URL, USER, PASSWORD);
    }
    public void createChatData(String input, String output) {
        String query = "INSERT INTO chatbot_data (user_question, bot_response) VALUES (?, ?)"; // Modified table and column names
        Connection connection = null;
        PreparedStatement stmt = null;
        try {
            connection = getConnection();
            stmt = connection.prepareStatement(query);
            stmt.setString(1, input);
            stmt.setString(2, output);
            stmt.executeUpdate();
        } catch (SQLException e) {
            e.printStackTrace();
        } finally {
            closeResources(connection, stmt, null);
        }
    }
    public List<String[]> readChatData() {
        List<String[]> dataList = new ArrayList<>();
        String query = "SELECT user_input, bot_response FROM chatbot_data"; // Modified column names
        Connection connection = null;
        PreparedStatement stmt = null;
        ResultSet rs = null;
        try {
            connection = getConnection();
            stmt = connection.prepareStatement(query);
            rs = stmt.executeQuery();
            while (rs.next()) {
                String[] data = new String[]{rs.getString("user_input"), rs.getString("bot_response")};
                dataList.add(data);
            }
        } catch (SQLException e) {
            e.printStackTrace();
        } finally {
            closeResources(connection, stmt, rs);
        }
        return dataList;
    }
    public void updateChatData(int id, String input, String output) {
        String query = "UPDATE chatbot_data SET user_input = ?, bot_response = ? WHERE id = ?"; // Modified column names
        Connection connection = null;
        PreparedStatement stmt = null;
        try {
            connection = getConnection();
            stmt = connection.prepareStatement(query);
            stmt.setString(1, input);
            stmt.setString(2, output);
            stmt.setInt(3, id);
            stmt.executeUpdate();
        } catch (SQLException e) {
            e.printStackTrace();
        } finally {
            closeResources(connection, stmt, null);
        }
    }
    public void deleteChatData(int id) throws SQLException {
        String query = "DELETE FROM chatbot_data WHERE id = ?"; // Modified table name
        Connection connection = null;
        PreparedStatement stmt = null;
        try {
            connection = getConnection();
            stmt = connection.prepareStatement(query);
            stmt.setInt(1, id);
            stmt.executeUpdate();
        } catch (SQLException e) {
            e.printStackTrace();
        } finally {
            closeResources(connection, stmt, null);
        }
    }
    private void closeResources(Connection connection, PreparedStatement stmt, ResultSet rs) {
        if (rs != null) {
            try {
                rs.close();
            } catch (SQLException e) {
                e.printStackTrace();
            }
        }
        if (stmt != null) {
            try {
                stmt.close();
            } catch (SQLException e) {
                e.printStackTrace();
            }
        }
        if (connection != null) {
            try {
                connection.close();
            } catch (SQLException e) {
                e.printStackTrace();
            }
        }
    }
}